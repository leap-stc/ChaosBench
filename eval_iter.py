import os
import datetime
import argparse
import yaml
import numpy as np
import pandas as pd
import xarray as xr
from collections import defaultdict as dd
import gc
from pathlib import Path
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
pl.seed_everything(42)

import sys
sys.path.append('..')

import warnings
warnings.filterwarnings('ignore')

from chaosbench import dataset, config, utils, criterion
from chaosbench.models import model


def main(args):
    """
    Evaluation script given .yaml config and trained model checkpoint (for iterative scheme)
    You can also provide a list from lra5 and oras5 if you want some or all of their parameters evaluated
        e.g., python eval_iter.py --model_name <model_name> --eval_years <eval_years> --lra5 [...] --oras5 [...]

    Example usage:
        (Climatology)              0) `python eval_iter.py --model_name climatology --eval_years 2022`
        (Persistence)              1) `python eval_iter.py --model_name persistence --eval_years 2022`
        (Physical models)          2) `python eval_iter.py --model_name ecmwf --eval_years 2022`
        (Data-driven models)       3) `python eval_iter.py --model_name unet_s2s --eval_years 2022 --version_num 0`
    
    """
    print(f'Evaluating reanalysis against {args.model_name}...')
    
    #########################################
    ####### Evaluation initialization #######
    #########################################
    
    IS_CLIMATOLOGY, IS_PERSISTENCE, IS_AI_MODEL = False, False, False
    BATCH_SIZE = 32
    
    ## Prepare directory to load model
    log_dir = Path('logs') / args.model_name
    ALL_PARAM_LIST = {'era5': utils.get_param_level_list(), 'lra5': config.LRA5_PARAMS, 'oras5': config.ORAS5_PARAMS}

    ## Case 0: Climatology
    if args.model_name == 'climatology':
        IS_CLIMATOLOGY = True
        PARAM_LIST = {'era5': utils.get_param_level_list(), 'lra5': config.LRA5_PARAMS, 'oras5': config.ORAS5_PARAMS}
        
        input_filepath = {
            'era5': os.path.join(config.DATA_DIR, 'climatology', 'climatology_era5.zarr'),
            'lra5': os.path.join(config.DATA_DIR, 'climatology', 'climatology_lra5.zarr'),
            'oras5': os.path.join(config.DATA_DIR, 'climatology', 'climatology_oras5.zarr'),
        }

        input_dataset = {
            'era5': xr.open_dataset(input_filepath['era5'], engine='zarr')['mean'].values,
            'lra5': xr.open_dataset(input_filepath['lra5'], engine='zarr')['mean'].values,
            'oras5': xr.open_dataset(input_filepath['oras5'], engine='zarr')['mean'].values,
        }
 
        # Retrieve all output files
        output_files = {'era5': [], 'lra5': [], 'oras5': []}
        for param_class, _ in output_files.items():
            
            output_filepath = os.path.join(config.DATA_DIR, param_class)
            pattern = rf'.*{year}\d{{4}}\.zarr$'
            
            for year in args.eval_years:
                output_files[param_class].extend(
                    [f"gs://{f}" for f in config.fs.ls(output_filepath) if re.match(pattern, str(f))]
                )
            
            output_files[param_class].sort()
        
        # Load all output data
        output_dataset = {'era5': [], 'lra5': [], 'oras5': []}
        for param_class, _ in output_dataset.items():
            for file_path in output_files[param_class]:
                ds = xr.open_dataset(file_path, engine='zarr')
                
                if param_class == 'era5':
                    output_dataset[param_class].append(ds[config.PARAMS].to_array().values.reshape(-1, 121, 240))
                else:
                    output_dataset[param_class].append(ds[ALL_PARAM_LIST[param_class]].to_array().values)
             
    ## Case 1: Persistence
    elif args.model_name == 'persistence':
        IS_PERSISTENCE = True
        PARAM_LIST = {'era5': utils.get_param_level_list(), 'lra5': config.LRA5_PARAMS, 'oras5': config.ORAS5_PARAMS}
        
        input_dataset = dataset.S2SObsDataset(
            years=args.eval_years, n_step=config.N_STEPS-1, 
            land_vars=config.LRA5_PARAMS, ocean_vars=config.ORAS5_PARAMS, is_normalized=False
        )
        input_dataloader = DataLoader(input_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        output_dataset = dataset.S2SObsDataset(
            years=args.eval_years, n_step=config.N_STEPS-1, 
            land_vars=config.LRA5_PARAMS, ocean_vars=config.ORAS5_PARAMS, is_normalized=False
        )
        output_dataloader = DataLoader(output_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
    
    ## Case 2: Physical models, one of [ecmwf, cma, ukmo, ncep]
    elif args.model_name in list(config.S2S_CENTERS.keys()):
        IS_NWPS = True
        PARAM_LIST = {'era5': utils.get_param_level_list(), 'lra5': args.lra5, 'oras5': args.oras5}

        input_dataset = dataset.S2SEvalDataset(s2s_name=args.model_name, years=args.eval_years, is_normalized=False)
        input_dataloader = DataLoader(input_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        output_dataset = dataset.S2SObsDataset(
            years=args.eval_years, n_step=config.N_STEPS-1, 
            land_vars=config.LRA5_PARAMS, ocean_vars=config.ORAS5_PARAMS, is_normalized=False
        )
        output_dataloader = DataLoader(output_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
    ## Case 3: Data-driven models
    elif 's2s' in args.model_name:
        IS_AI_MODEL = True
        PARAM_LIST = {'era5': utils.get_param_level_list(), 'lra5': args.lra5, 'oras5': args.oras5}

        ## Retrieve hyperparameters
        config_filepath = Path(f'chaosbench/configs/{args.model_name}.yaml')
        with open(config_filepath, 'r') as config_f:
            hyperparams = yaml.load(config_f, Loader=yaml.FullLoader)

        model_args = hyperparams['model_args']
        data_args = hyperparams['data_args']
        
        ## Initialize model given hyperparameters
        baseline = model.S2SBenchmarkModel(model_args=model_args, data_args=data_args)
        
        ## Load model from checkpoint
        ckpt_filepath = log_dir / f'lightning_logs/version_{args.version_num}/checkpoints/'
        ckpt_filepath = list(ckpt_filepath.glob('*.ckpt'))[0]
        baseline = baseline.load_from_checkpoint(ckpt_filepath)
        baseline.eval()
        
        lra5_vars = baseline.hparams.get('data_args', {}).get('land_vars', [])
        oras5_vars = baseline.hparams.get('data_args', {}).get('ocean_vars', [])

        ## Prepare input/output dataset
        input_dataset = dataset.S2SObsDataset(args.eval_years, config.N_STEPS-1, land_vars=lra5_vars, ocean_vars=oras5_vars)
        input_dataloader = DataLoader(input_dataset, batch_size=BATCH_SIZE, shuffle=False)

        output_dataset = dataset.S2SObsDataset(
            args.eval_years, config.N_STEPS-1, 
            land_vars=config.LRA5_PARAMS, ocean_vars=config.ORAS5_PARAMS, is_normalized=False
        )
        output_dataloader = DataLoader(output_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
    else:
        raise NotImplementedError('Inference type has yet to be implemented...')
        
    
    ####################### Initialize criteria ###################
    RMSE = criterion.RMSE()
    Bias = criterion.Bias()
    ACC = criterion.ACC()
    SSIM = criterion.MS_SSIM()
    SpecDiv = criterion.SpectralDiv(percentile=0.9, is_train=False)
    SpecRes = criterion.SpectralRes(percentile=0.9, is_train=False)
    ###############################################################
    
    
    ######################################
    ####### Evaluation main script #######
    ######################################
    
    
    ## All metric placeholders
    all_rmse, all_bias, all_acc, all_ssim, all_sdiv, all_sres = list(), list(), list(), list(), list(), list()
    
    if IS_CLIMATOLOGY:
        
        ## Step metric placeholders
        step_rmse, step_bias, step_acc, step_ssim, step_sdiv, step_sres = dict(), dict(), dict(), dict(), dict(), dict()
        
        # Pre-processing (e.g., get day-of-years for climatology-related metrics...)
        doys = [doy for year in args.eval_years for doy in range(1, 367 if datetime.date(int(year), 12, 31).timetuple().tm_yday == 366 else 366)]

        ## Extract metric for each param/level
        for i, (param_class, params) in enumerate(ALL_PARAM_LIST.items()):

            input_dataset[param_class] = torch.tensor(input_dataset[param_class])
            output_dataset[param_class] = torch.tensor(output_dataset[param_class])

            for j, param in enumerate(params):

                ### Some param/level pairs are not available
                param_exist = param in PARAM_LIST[param_class]

                ## Handling predictions
                unique_preds = input_dataset[param_class][:output_dataset[param_class].shape[0], j]
                unique_preds = unique_preds.double().to(config.device)
                
                ## Handling labels
                unique_labels = output_dataset[param_class][:, j]
                unique_labels = unique_labels.double().to(config.device)
                
                ################################## Criterion 1: RMSE #####################################
                error = RMSE(unique_preds, unique_labels).cpu().numpy()

                ################################## Criterion 2: Bias #####################################
                bias = Bias(unique_preds, unique_labels).cpu().numpy()

                ################################## Criterion 3: ACC ######################################
                acc = ACC(unique_preds, unique_labels, doys, param, param_class).cpu().numpy()

                ################################## Criterion 4: SSIM ######################################
                ssim = SSIM(unique_preds, unique_labels).cpu().numpy()

                ################################ Criterion 5: SpecDiv #####################################
                sdiv = SpecDiv(unique_preds, unique_labels).cpu().numpy()

                ################################ Criterion 6: SpecRes #####################################
                sres = SpecRes(unique_preds, unique_labels).cpu().numpy()

                step_rmse[param] = [error] * (config.N_STEPS - 1)
                step_bias[param] = [bias] * (config.N_STEPS - 1)
                step_acc[param] = [acc] * (config.N_STEPS - 1)
                step_ssim[param] = [ssim] * (config.N_STEPS - 1)
                step_sdiv[param] = [sdiv] * (config.N_STEPS - 1)
                step_sres[param] = [sres] * (config.N_STEPS - 1)

        all_rmse.append(step_rmse)
        all_bias.append(step_bias)
        all_acc.append(step_acc)
        all_ssim.append(step_ssim)
        all_sdiv.append(step_sdiv)
        all_sres.append(step_sres)
                        
    
    else:
        
        for input_batch, output_batch in tqdm(zip(input_dataloader, output_dataloader), total=len(input_dataloader)):

            _, preds_x, preds_y = input_batch
            timestamps, truth_x, truth_y = output_batch
            
            # Pre-processing (e.g., get day-of-years for climatology-related metrics...)
            doys = utils.get_doys_from_timestep(timestamps)
           
            ## Step metric placeholders
            step_rmse, step_bias, step_acc, step_ssim, step_sdiv, step_sres = dict(), dict(), dict(), dict(), dict(), dict()

            for step_idx in range(truth_y.size(1)):
                all_param_idx, param_idx = 0, 0

                ############## Getting current-step target ##############
                targs = truth_y[:, step_idx]
                #########################################################

                with torch.no_grad():

                    ############# Getting current-step preds #############
                    if IS_AI_MODEL:
                        preds = baseline(preds_x.to(config.device))

                    elif IS_PERSISTENCE:
                        preds = preds_x

                    else:
                        preds = preds_y[:, step_idx]
                    #####################################################


                    ## Compute metric for each parameter
                    for i, (param_class, params) in enumerate(ALL_PARAM_LIST.items()):
                        for j, param in enumerate(params):
                            
                            ### Some param/level pairs are not available 
                            param_exist = param in PARAM_LIST[param_class]

                            ## Handling predictions
                            unique_preds = preds[:, param_idx] if param_exist else torch.full((BATCH_SIZE, 121, 240), torch.nan)
                            unique_preds = utils.denormalize(unique_preds, param, param_class) if IS_AI_MODEL else unique_preds
                            unique_preds = unique_preds.double().to(config.device)
                            
                            ## Handling labels
                            unique_labels = targs[:, all_param_idx]
                            unique_labels = unique_labels.double().to(config.device)

                            ################################## Criterion 1: RMSE #####################################
                            error = RMSE(unique_preds, unique_labels).cpu().numpy()

                            ################################## Criterion 2: Bias #####################################
                            bias = Bias(unique_preds, unique_labels).cpu().numpy()

                            ################################## Criterion 3: ACC ######################################
                            acc = ACC(unique_preds, unique_labels, doys[:, step_idx], param, param_class).cpu().numpy()

                            ################################## Criterion 4: SSIM ######################################
                            ssim = SSIM(unique_preds, unique_labels).cpu().numpy()

                            ################################ Criterion 5: SpecDiv #####################################
                            sdiv = SpecDiv(unique_preds, unique_labels).cpu().numpy()

                            ################################ Criterion 6: SpecRes #####################################
                            sres = SpecRes(unique_preds, unique_labels).cpu().numpy()


                            try:
                                step_rmse[param].extend([error])
                                step_bias[param].extend([bias])
                                step_acc[param].extend([acc])
                                step_ssim[param].extend([ssim])
                                step_sdiv[param].extend([sdiv])
                                step_sres[param].extend([sres])

                            except:
                                step_rmse[param] = [error]
                                step_bias[param] = [bias]
                                step_acc[param] = [acc]
                                step_ssim[param] = [ssim]
                                step_sdiv[param] = [sdiv]
                                step_sres[param] = [sres]

                            all_param_idx += 1
                            param_idx = param_idx + 1 if param_exist else param_idx

                    ## Make next-step input as the current prediction (used for AI models)
                    preds_x = preds

                ## (1) Cleaning up to release memory at each time_step
                targs, preds = targs.cpu().detach(), preds.cpu().detach()
                del targs, preds

            ## (2) Cleaning up to release memory at each batch
            preds_x, preds_y = preds_x.cpu().detach(), preds_y.cpu().detach()
            truth_x, truth_y = truth_x.cpu().detach(), truth_y.cpu().detach()
            del preds_x, preds_y, truth_x, truth_y
            gc.collect()

            all_rmse.append(step_rmse)
            all_bias.append(step_bias)
            all_acc.append(step_acc)
            all_ssim.append(step_ssim)
            all_sdiv.append(step_sdiv)
            all_sres.append(step_sres)
    
    ## Combine metrics across batch
    merged_rmse, merged_bias, merged_acc, \
    merged_ssim, merged_sdiv, merged_sres = dd(list), dd(list), dd(list), dd(list), dd(list), dd(list)
    
    for d_rmse, d_bias, d_acc, d_ssim, d_sdiv, d_sres in zip(all_rmse, all_bias, all_acc, all_ssim, all_sdiv, all_sres):
        for (rmse_k, rmse_v), (bias_k, bias_v), (acc_k, acc_v), \
            (ssim_k, ssim_v), (sdiv_k, sdiv_v), (sres_k, sres_v) in zip(d_rmse.items(),
                                                                        d_bias.items(),
                                                                        d_acc.items(),
                                                                        d_ssim.items(),
                                                                        d_sdiv.items(),
                                                                        d_sres.items()):
            
            merged_rmse[rmse_k].append(rmse_v)
            merged_bias[bias_k].append(bias_v)
            merged_acc[acc_k].append(acc_v)
            merged_ssim[ssim_k].append(ssim_v)
            merged_sdiv[sdiv_k].append(sdiv_v)
            merged_sres[sres_k].append(sres_v)

    ## Compute the mean metrics over valid evaluation time horizon (for each timestep) along batch
    merged_rmse, \
    merged_bias, \
    merged_acc, \
    merged_ssim, \
    merged_sdiv, \
    merged_sres = dict(merged_rmse), dict(merged_bias), dict(merged_acc), dict(merged_ssim), dict(merged_sdiv), dict(merged_sres)
    
    for (rmse_k, rmse_v), (bias_k, bias_v), (acc_k, acc_v), \
        (ssim_k, ssim_v), (sdiv_k, sdiv_v), (sres_k, sres_v) in zip(merged_rmse.items(), 
                                                                    merged_bias.items(),
                                                                    merged_acc.items(),
                                                                    merged_ssim.items(),
                                                                    merged_sdiv.items(),
                                                                    merged_sres.items()):
        
        merged_rmse[rmse_k] = np.array(merged_rmse[rmse_k]).mean(axis=0)
        merged_bias[bias_k] = np.array(merged_bias[bias_k]).mean(axis=0)
        merged_acc[acc_k] = np.array(merged_acc[acc_k]).mean(axis=0)
        merged_ssim[ssim_k] = np.array(merged_ssim[ssim_k]).mean(axis=0)
        merged_sdiv[sdiv_k] = np.array(merged_sdiv[sdiv_k]).mean(axis=0)
        merged_sres[sres_k] = np.array(merged_sres[sres_k]).mean(axis=0)
        
    ## Save metrics
    eval_dir = log_dir / 'eval' / f'version_{args.version_num}' if IS_AI_MODEL else log_dir / 'eval'
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame(merged_rmse).to_csv(eval_dir / f'rmse_{args.model_name}.csv', index=False)
    pd.DataFrame(merged_bias).to_csv(eval_dir / f'bias_{args.model_name}.csv', index=False)
    pd.DataFrame(merged_acc).to_csv(eval_dir / f'acc_{args.model_name}.csv', index=False)
    pd.DataFrame(merged_ssim).to_csv(eval_dir / f'ssim_{args.model_name}.csv', index=False)
    pd.DataFrame(merged_sdiv).to_csv(eval_dir / f'sdiv_{args.model_name}.csv', index=False)
    pd.DataFrame(merged_sres).to_csv(eval_dir / f'sres_{args.model_name}.csv', index=False)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help='Name of the model specified in your config file, including one of [ecmwf, cma, ukmo, ncep, persistence]')
    parser.add_argument('--eval_years', nargs='+', help='Provide the evaluation years')
    parser.add_argument('--version_num', default=0, help='Version number of the model')
    parser.add_argument('--lra5', nargs='+', type=str, default=[], help='List of LRA5 variables to be evaluated')
    parser.add_argument('--oras5', nargs='+', type=str, default=[], help='List of ORAS5 variables to be evaluated')

    args = parser.parse_args()
    main(args)
