import os
import argparse
import re
import yaml
import numpy as np
import pandas as pd
import xarray as xr
import cfgrib
from collections import defaultdict as dd
from datetime import datetime
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
    Evaluation script to assess data-driven models
    See `scripts/process_sota.py` on how to run forecasts on these models, including pangu, fcn2, graphcast
    
    Example usage:
        (Panguweather)             1) `python eval_sota.py --model_name panguweather --eval_years 2022`
        (Graphcast)                2) `python eval_sota.py --model_name graphcast --eval_years 2022`
        (FourcastNetV2)            3) `python eval_sota.py --model_name fourcastnetv2 --eval_years 2022`
    
    """
    assert args.model_name in ['panguweather', 'fourcastnetv2', 'graphcast']
    print(f'Evaluating reanalysis against {args.model_name}...')
    
    #########################################
    ####### Evaluation initialization #######
    #########################################
    
    ## Prepare directory to load model
    log_dir = Path('logs') / args.model_name
    ALL_PARAM_LIST = {'era5': utils.get_param_level_list(), 'lra5': config.LRA5_PARAMS, 'oras5': config.ORAS5_PARAMS}
    PARAM_LIST = {'era5': [], 'lra5': [], 'oras5': []}
    
    ## Get prediction
    input_dataset = sorted([
        f for year in args.eval_years 
        for f in (config.DATA_DIR / args.model_name).glob(f'*{year}*.zarr')
        if re.match(rf'.*{year}\d{{4}}\.zarr$', f.name)
    ])
    
    ## Get target
    target_dataset = dataset.S2SObsDataset(
        years=args.eval_years, n_step=config.N_STEPS-1, 
        land_vars=config.LRA5_PARAMS, ocean_vars=config.ORAS5_PARAMS, is_normalized=False
    )
        
    ####################### Initialize criteria #######################
    RMSE = criterion.RMSE()
    Bias = criterion.Bias()
    ACC = criterion.ACC()
    SSIM = criterion.MS_SSIM()
    SpecDiv = criterion.SpectralDiv(percentile=0.9, is_train=False)
    SpecRes = criterion.SpectralRes(percentile=0.9, is_train=False)
    ##################################################################
    
    
    ######################################
    ####### Evaluation main script #######
    ######################################
    
    ## All metric placeholders
    all_rmse, all_bias, all_acc, all_ssim, all_sdiv, all_sres = list(), list(), list(), list(), list(), list()
    
    for input_d in tqdm(input_dataset):

        preds_dataset = xr.open_dataset(input_d, engine='zarr')
        PARAM_LIST['era5'] = list(preds_dataset.data_vars)
        
        # Pre-processing (e.g., get day-of-years for climatology-related metrics...)
        doy = int(pd.to_datetime(str(preds_dataset.time.values)).dayofyear)

        if doy <= len(target_dataset):
            timestamps, truth_x, truth_y = target_dataset[doy - 1]
            doys = utils.get_doys_from_timestep(torch.tensor([timestamps])) ## Batch-level DOYs

            ## Step metric placeholders
            step_rmse, step_bias, step_acc, step_ssim, step_sdiv, step_sres = dict(), dict(), dict(), dict(), dict(), dict()

            for step_idx in range(truth_y.size(0)):
                param_idx = 0

                ############## Getting current-step preds/targs ##############
                preds = preds_dataset.isel(step=step_idx)
                targs = truth_y[step_idx]
                ##############################################################

                ## Compute metric for each parameter
                for i, (param_class, params) in enumerate(ALL_PARAM_LIST.items()):
                    for j, param in enumerate(params):

                        ### Some param/level pairs are not available 
                        param_exist = param in PARAM_LIST[param_class]

                        ## Handling predictions
                        unique_preds = torch.tensor(preds[param].values) if param_exist else torch.full((121,240), torch.nan) 
                        unique_preds = unique_preds.double().unsqueeze(0).to(config.device)

                        ## Handling labels (land/ocean masking)
                        unique_labels = targs[param_idx] if param_class == 'era5' else torch.where(targs[param_idx] == 0.0, torch.nan, targs[param_idx])
                        unique_labels = unique_labels.double().unsqueeze(0).to(config.device)

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
                        
                        param_idx += 1
                                
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
    eval_dir = log_dir / 'eval'
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame(merged_rmse).to_csv(eval_dir / f'rmse_{args.model_name}.csv', index=False)
    pd.DataFrame(merged_bias).to_csv(eval_dir / f'bias_{args.model_name}.csv', index=False)
    pd.DataFrame(merged_acc).to_csv(eval_dir / f'acc_{args.model_name}.csv', index=False)
    pd.DataFrame(merged_ssim).to_csv(eval_dir / f'ssim_{args.model_name}.csv', index=False)
    pd.DataFrame(merged_sdiv).to_csv(eval_dir / f'sdiv_{args.model_name}.csv', index=False)
    pd.DataFrame(merged_sres).to_csv(eval_dir / f'sres_{args.model_name}.csv', index=False)
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help='Name of the model specified in your config file, including one of [fourcastnet, fourcastnetv2, panguweather, graphcast]')
    parser.add_argument('--eval_years', nargs='+', help='Provide the evaluation years')
    args = parser.parse_args()
    main(args)
