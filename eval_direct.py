import os
import argparse
import yaml
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from collections import defaultdict as dd
import copy
import gc
import pickle
from pathlib import Path
from tqdm import tqdm
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

def interpolate(values, indices):
    """Interpolate in between values if delta_t is not dense"""
    # First, we check if there are NaN values (i.e., no param/level combination in prediction)
    has_nan = np.any(np.isnan(values)) 
    new_indices = np.arange(indices[0], indices[-1])
    
    ## If yes, fill everything with nan
    if has_nan:
        new_values = np.full(len(new_indices), np.nan)
    
    ## Otherwise, interpolate
    else:
        f = interp1d(indices, values, bounds_error=False)
        new_values = f(new_indices)
    
    return new_values
    

def main(args):
    """
    Evaluation script given .yaml config and trained model checkpoint (for direct scheme)
    You can also provide a list from lra5 and oras5 if you want some or all of their parameters predicted, 
        e.g., python eval_direct.py --model_name <model_name> --eval_years <eval_years> --version_nums [...] --task_num <task_num> --lra5 [...] --oras5 [...]

    Example usage:
    (Data-driven models)       
    1) `python eval_direct.py --model_name unet_s2s --eval_years 2022 --version_nums 0 4 5 6 7 8 9 10 11 12 --task_num 1`
    2) `python eval_direct.py --model_name unet_s2s --eval_years 2022 --version_nums 2 13 14 15 16 17 18 19 20 21 --task_num 2`
    
    (External predictions)     
    3) `python eval_direct.py --model_name climax --eval_years 2022 --task_num 1`
    
    """
    assert args.task_num in [1, 2]
    
    print(f'Evaluating reanalysis against {args.model_name}...')
    
    #########################################
    ####### Evaluation initialization #######
    #########################################
    
    IS_AI_MODEL, IS_PREDICTION = False, False
    BATCH_SIZE = 32
    DELTA_T = np.array([1, 5, 10, 15, 20, 25, 30, 35, 40, 44])
    
    ## Prepare directory to load model
    log_dir = Path('logs') / args.model_name
    ALL_PARAM_LIST = {'era5': utils.get_param_level_list(), 'lra5': config.LRA5_PARAMS, 'oras5': config.ORAS5_PARAMS}
    
    ## Case 1: Data-driven model
    if 's2s' in args.model_name:
        IS_AI_MODEL = True
        PARAM_LIST = {'era5': utils.get_param_level_list(), 'lra5': args.lra5, 'oras5': args.oras5}
        
        ## Retrieve hyperparameters
        config_filepath = Path(f'chaosbench/configs/{args.model_name}.yaml')
        with open(config_filepath, 'r') as config_f:
            hyperparams = yaml.load(config_f, Loader=yaml.FullLoader)
        
        model_args = hyperparams['model_args']
        data_args = hyperparams['data_args']

        ## Initialize model given hyperparameters
        assert len(args.version_nums) == len(DELTA_T)
        
        ## Load each model from checkpoint
        baselines = list()
        for version_num in args.version_nums:
            ckpt_filepath = log_dir / f'lightning_logs/version_{version_num}/checkpoints/'
            ckpt_filepath = list(ckpt_filepath.glob('*.ckpt'))[0]
            baseline = model.S2SBenchmarkModel(model_args=model_args, data_args=data_args)
            baseline = baseline.load_from_checkpoint(ckpt_filepath)
            baselines.append(copy.deepcopy(baseline.eval()))
        
        ## Prepare input/output dataset
        lra5_vars, oras5_vars = baseline.hparams.get('land_vars', []), baseline.hparams.get('ocean_vars', [])
        input_dataset = dataset.S2SObsDataset(
            years=args.eval_years, n_step=config.N_STEPS-1, land_vars=lra5_vars, ocean_vars=oras5_vars
        )
        input_dataloader = DataLoader(input_dataset, batch_size=BATCH_SIZE, shuffle=False)

        output_dataset = dataset.S2SObsDataset(
            years=args.eval_years, n_step=config.N_STEPS-1, 
            land_vars=config.LRA5_PARAMS, ocean_vars=config.ORAS5_PARAMS, is_normalized=False
        )
        output_dataloader = DataLoader(output_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
    
    ## Case 2: External prediction (e.g., ClimaX)
    else:
        IS_EXTERNAL = True
        PARAM_LIST = {'era5': config.CLIMAX_VARS if args.task_num == 1 else config.HEADLINE_VARS, 'lra5': args.lra5, 'oras5': args.oras5}
        
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
        
        ## List external prediction
        preds_filepath = log_dir / 'preds' / f'task{args.task_num}'
        preds_files = list(preds_filepath.glob('*.pkl'))
        preds_files.sort()
        
        ## Load prediction
        all_preds = list()
        for file_path in preds_files:
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                data = data['pred']
                all_preds.append(data)
                
        all_preds = np.array(all_preds)

    ##################### Initialize criteria #####################
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
    batch_idx = 0

    for input_batch, output_batch in tqdm(zip(input_dataloader, output_dataloader), total=len(input_dataloader)):
        
        _, preds_x, preds_y = input_batch
        timestamps, truth_x, truth_y = output_batch

        # Pre-processing (e.g., get day-of-years for climatology-related metrics...)
        doys = utils.get_doys_from_timestep(timestamps)
        
        assert preds_y.size(1) == truth_y.size(1)
        N_STEPS = truth_y.size(1)
        
        ## Step metric placeholders
        step_rmse, step_bias, step_acc, step_ssim, step_sdiv, step_sres = dict(), dict(), dict(), dict(), dict(), dict()

        for step_idx, delta in enumerate(DELTA_T):
            all_param_idx, param_idx = 0, 0

            with torch.no_grad(): 
                
                ############## Getting current-step preds/targs ##############
                if IS_AI_MODEL:
                    preds = baselines[step_idx](preds_x.to(config.device))
                    targs = truth_y[:, delta-1]
                
                else:
                    preds = all_preds[step_idx]
                    targs = truth_y[:, delta-1]
                ##############################################################
                
                
                ## Extract metric for each param/level
                for i, (param_class, params) in enumerate(ALL_PARAM_LIST.items()):
                    for j, param in enumerate(params):
                        
                        ### Some param/level pairs are not available 
                        param_exist = param in PARAM_LIST[param_class]

                        ## Handling predictions
                        if IS_AI_MODEL: 
                            unique_preds = preds[:, param_idx] if param_exist else torch.full((BATCH_SIZE, 121, 240), torch.nan)
                            unique_preds = utils.denormalize(unique_preds, param, param_class)
                            unique_preds = unique_preds.double().to(config.device)
                        
                        else:
                            param_name, start_idx, end_idx = param.replace('-', '_'), int(batch_idx * BATCH_SIZE), int((batch_idx + 1) * BATCH_SIZE)
                            unique_preds = torch.tensor(preds[param_name])[start_idx:end_idx] if param_exist else torch.full((BATCH_SIZE, 121, 240), torch.nan)
                            unique_preds = unique_preds.double().to(config.device)
                        
                        ## Handling labels
                        unique_labels = targs[:, all_param_idx]
                        unique_labels = unique_labels.double().to(config.device)
                        
                        ################################## Criterion 1: RMSE #####################################
                        error = RMSE(unique_preds, unique_labels).cpu().numpy()

                        ################################## Criterion 2: Bias #####################################
                        bias = Bias(unique_preds, unique_labels).cpu().numpy()
                        
                        ################################## Criterion 3: ACC ######################################
                        acc = ACC(unique_preds, unique_labels, doys[:, delta-1], param, param_class).cpu().numpy()
                        
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
                        
        all_rmse.append(step_rmse)
        all_bias.append(step_bias)
        all_acc.append(step_acc)
        all_ssim.append(step_ssim)
        all_sdiv.append(step_sdiv)
        all_sres.append(step_sres)
        
        batch_idx += 1 # keeping track of batch_id to subset prediction index
    
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
    ## Also interpolate given gaps in delta_t
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
        
        merged_rmse[rmse_k] = interpolate(np.array(merged_rmse[rmse_k]).mean(axis=0), DELTA_T - 1)
        merged_bias[bias_k] = interpolate(np.array(merged_bias[bias_k]).mean(axis=0), DELTA_T - 1)
        merged_acc[acc_k] = interpolate(np.array(merged_acc[acc_k]).mean(axis=0), DELTA_T - 1)
        merged_ssim[ssim_k] = interpolate(np.array(merged_ssim[ssim_k]).mean(axis=0), DELTA_T - 1)
        merged_sdiv[sdiv_k] = interpolate(np.array(merged_sdiv[sdiv_k]).mean(axis=0), DELTA_T - 1)
        merged_sres[sres_k] = interpolate(np.array(merged_sres[sres_k]).mean(axis=0), DELTA_T - 1)
        
        
    ## Save metrics
    eval_dir = log_dir / 'eval' / f'direct_{args.task_num}'
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame(merged_rmse).to_csv(eval_dir / f'rmse_{args.model_name}.csv', index=False)
    pd.DataFrame(merged_bias).to_csv(eval_dir / f'bias_{args.model_name}.csv', index=False)
    pd.DataFrame(merged_acc).to_csv(eval_dir / f'acc_{args.model_name}.csv', index=False)
    pd.DataFrame(merged_ssim).to_csv(eval_dir / f'ssim_{args.model_name}.csv', index=False)
    pd.DataFrame(merged_sdiv).to_csv(eval_dir / f'sdiv_{args.model_name}.csv', index=False)
    pd.DataFrame(merged_sres).to_csv(eval_dir / f'sres_{args.model_name}.csv', index=False)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help='Name of the model either specified in your config file or external')
    parser.add_argument('--eval_years', nargs='+', help='Provide the evaluation years')
    parser.add_argument('--version_nums', nargs='+', help='Provide the version numbers')
    parser.add_argument('--task_num', type=int, default=1, help='Task number, one of [1,2]')
    parser.add_argument('--lra5', nargs='+', type=str, default=[], help='List of LRA5 variables to be evaluated')
    parser.add_argument('--oras5', nargs='+', type=str, default=[], help='List of ORAS5 variables to be evaluated')

    args = parser.parse_args()
    main(args)
