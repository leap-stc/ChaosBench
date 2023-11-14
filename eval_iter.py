import os
import argparse
import yaml
import numpy as np
import pandas as pd
import xarray as xr
from collections import defaultdict as dd
import gc
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

def main(args):
    """
    Evaluation script given .yaml config and trained model checkpoint (for iterative scheme)
    
    Example usage:
        (Climatology)              0) `python eval_iter.py --model_name climatology --eval_years 2022`
        (Persistence)              1) `python eval_iter.py --model_name persistence --eval_years 2022`
        (Physical models)          2) `python eval_iter.py --model_name ecmwf --eval_years 2022`
        (Data-driven models)       3) `python eval_iter.py --model_name mlp_s2s --eval_years 2022 --version_num 0`
    
    """
    print(f'Evaluating ERA5 observations against {args.model_name}...')
    
    #########################################
    ####### Evaluation initialization #######
    #########################################
    
    IS_AI_MODEL, IS_PERSISTENCE, IS_CLIMATOLOGY = False, False, False
    BATCH_SIZE = 32
    
    ## Prepare directory to load model
    log_dir = Path('logs') / args.model_name
    
    ## Case 0: Climatology
    if args.model_name == 'climatology':
        IS_CLIMATOLOGY = True
        input_dataset = dataset.S2SObsDataset(years=args.eval_years, n_step=config.N_STEPS-1)
        input_dataloader = DataLoader(input_dataset, batch_size=BATCH_SIZE, shuffle=False)
        output_dataset = dataset.S2SObsDataset(years=args.eval_years, n_step=config.N_STEPS-1)
        output_dataloader = DataLoader(output_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        climatology_filepath = Path(config.DATA_DIR) / 's2s' / 'climatology' / 'climatology_era5_spatial.zarr'
        climatology = xr.open_dataset(climatology_filepath, engine='zarr')
        climatology = climatology['mean'].values[np.newaxis, :, :, :, :] # (B, P, L, H, W)
    
    ## Case 1: Persistence
    elif args.model_name == 'persistence':
        IS_PERSISTENCE = True
        input_dataset = dataset.S2SObsDataset(years=args.eval_years, n_step=config.N_STEPS-1)
        input_dataloader = DataLoader(input_dataset, batch_size=BATCH_SIZE, shuffle=False)
        output_dataset = dataset.S2SObsDataset(years=args.eval_years, n_step=config.N_STEPS-1)
        output_dataloader = DataLoader(output_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
    
    ## Case 2: Physical models, one of [ecmwf, cma, ukmo, ncep]
    elif args.model_name in list(config.S2S_CENTERS.keys()):
        input_dataset = dataset.S2SEvalDataset(s2s_name=args.model_name, years=args.eval_years)
        input_dataloader = DataLoader(input_dataset, batch_size=BATCH_SIZE, shuffle=False)
        output_dataset = dataset.S2SObsDataset(years=args.eval_years, n_step=config.N_STEPS-1)
        output_dataloader = DataLoader(output_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
    ## Case 3: Data-driven models
    else:
        IS_AI_MODEL = True
        
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
        
        ## Prepare input/output dataset
        input_dataset = dataset.S2SObsDataset(years=args.eval_years, n_step=config.N_STEPS-1)
        input_dataloader = DataLoader(input_dataset, batch_size=BATCH_SIZE, shuffle=False)
        output_dataset = dataset.S2SObsDataset(years=args.eval_years, n_step=config.N_STEPS-1)
        output_dataloader = DataLoader(output_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
    
    ##### Initialize criteria #####
    RMSE = criterion.RMSE()
    Bias = criterion.Bias()
    ACC = criterion.ACC()
    SSIM = criterion.MS_SSIM()
    SpecDiv = criterion.SpectralDiv(percentile=0.9)
    SpecRes = criterion.SpectralRes(percentile=0.9)
    ###############################
    
    
    ######################################
    ####### Evaluation main script #######
    ######################################
    
    ## All metric placeholders
    all_rmse, all_bias, all_acc, all_ssim, all_sdiv, all_sres = list(), list(), list(), list(), list(), list()

    for input_batch, output_batch in tqdm(zip(input_dataloader, output_dataloader), total=len(input_dataloader)):
        
        _, preds_x, preds_y = input_batch
        _, truth_x, truth_y = output_batch
        
        assert preds_y.size(1) == truth_y.size(1)
        
        curr_x = preds_x.to(config.device) # Initialize current x

        ## Step metric placeholders
        step_rmse, step_bias, step_acc, step_ssim, step_sdiv, step_sres = dict(), dict(), dict(), dict(), dict(), dict()

        for step_idx in range(truth_y.size(1)):

            with torch.no_grad():
                curr_y = truth_y[:, step_idx].to(config.device)
                
                ############## Getting predictions ##############
                if IS_AI_MODEL:
                    preds = baseline(curr_x)
                
                elif IS_PERSISTENCE:
                    preds = preds_x.to(config.device)
                    
                elif IS_CLIMATOLOGY:
                    preds = torch.tensor(climatology).to(config.device)
                
                else:
                    preds = preds_y[:, step_idx].to(config.device)
                #################################################
                

                ## Extract metric for each param/level
                for param_idx, param in enumerate(config.PARAMS):
                    for level_idx, level in enumerate(config.PRESSURE_LEVELS):

                        ## Handling predictions
                        if IS_CLIMATOLOGY:
                            unique_preds = preds[:, param_idx, level_idx, :, :]
                            unique_preds = unique_preds.repeat(BATCH_SIZE, 1, 1)
                            
                        else:
                            preds[:, param_idx, level_idx] = utils.denormalize(
                                preds[:, param_idx, level_idx], param, level, args.model_name
                            )
                            unique_preds = preds[:, param_idx, level_idx]
                        
                        ## Handling labels
                        curr_y[:, param_idx, level_idx] = utils.denormalize(
                            curr_y[:, param_idx, level_idx], param, level, 'era5'
                        )
                        unique_labels = curr_y[:, param_idx, level_idx]

                        ################################## Criterion 1: RMSE #####################################
                        error = RMSE(unique_preds, unique_labels).cpu().numpy()

                        ################################## Criterion 2: Bias #####################################
                        bias = Bias(unique_preds, unique_labels).cpu().numpy()
                        
                        ################################## Criterion 3: ACC ######################################
                        acc = ACC(unique_preds, unique_labels, param, level).cpu().numpy()
                        
                        ################################## Criterion 4: SSIM ######################################
                        ssim = SSIM(unique_preds, unique_labels).cpu().numpy()
                        
                        ################################ Criterion 5: SpecDiv #####################################
                        sdiv = SpecDiv(unique_preds, unique_labels).cpu().numpy()
                        
                        ################################ Criterion 6: SpecRes #####################################
                        sres = SpecRes(unique_preds, unique_labels).cpu().numpy()


                        try:
                            step_rmse[f'{param}-{level}'].extend([error])
                            step_bias[f'{param}-{level}'].extend([bias])
                            step_acc[f'{param}-{level}'].extend([acc])
                            step_ssim[f'{param}-{level}'].extend([ssim])
                            step_sdiv[f'{param}-{level}'].extend([sdiv])
                            step_sres[f'{param}-{level}'].extend([sres])

                        except:
                            step_rmse[f'{param}-{level}'] = [error]
                            step_bias[f'{param}-{level}'] = [bias]
                            step_acc[f'{param}-{level}'] = [acc]
                            step_ssim[f'{param}-{level}'] = [ssim]
                            step_sdiv[f'{param}-{level}'] = [sdiv]
                            step_sres[f'{param}-{level}'] = [sres]

                ## Make next-step input as the current prediction (used for AI models)
                curr_x = preds

            ## (1) Cleaning up to release memory at each time_step
            curr_y, preds = curr_y.cpu().detach(), preds.cpu().detach()
            del curr_y, preds
                    
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
    
    args = parser.parse_args()
    main(args)
