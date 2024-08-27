import os
import copy
import logging
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
    Evaluation script given .yaml config and trained model checkpoint (for iterative scheme) for ensemble forecasts
    You can also provide a list from lra5 and oras5 if you want some or all of their parameters predicted, 
        e.g., python eval_ensemble.py --model_name <model_name> --eval_years <eval_years> --lra5 [...] --oras5 [...]

    Example usage:
    (Physical models)          
    1) `python eval_ensemble.py --model_name cma --eval_years 2022`
    
    (Data-driven models: multiple checkpoints ensemble)       
    2) `python eval_ensemble.py --model_name unet_ensemble_s2s --eval_years 2022 --version_nums 0 1 2 3 4`
    
    """
    print(f'Evaluating reanalysis against {args.model_name}...')
    
    #########################################
    ####### Evaluation initialization #######
    #########################################
    
    IS_NWPS, IS_AI_MODEL = False, False
    BATCH_SIZE = 32
    
    ALL_PARAM_LIST = {'era5': utils.get_param_level_list(), 'lra5': config.LRA5_PARAMS, 'oras5': config.ORAS5_PARAMS}

    ## Case 1: Physical models, one of [ecmwf, cma, ukmo, ncep]
    if args.model_name in list(config.S2S_CENTERS.keys()):
        IS_NWPS = True
        log_dir = Path('logs') / f'{args.model_name}_ensemble'
        PARAM_LIST = {'era5': utils.get_param_level_list(), 'lra5': args.lra5, 'oras5': args.oras5}

        input_dataset = dataset.S2SEvalDataset(s2s_name=args.model_name, years=args.eval_years, is_ensemble=True, is_normalized=False)
        input_dataloader = DataLoader(input_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        output_dataset = dataset.S2SObsDataset(
            years=args.eval_years, n_step=config.N_STEPS-1, 
            land_vars=config.LRA5_PARAMS, ocean_vars=config.ORAS5_PARAMS, is_normalized=False
        )
        output_dataloader = DataLoader(output_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
    ## Case 2: Data-driven models
    elif 's2s' in args.model_name:
        IS_AI_MODEL = True
        log_dir = Path('logs') / f'{args.model_name}'
        PARAM_LIST = {'era5': utils.get_param_level_list(), 'lra5': args.lra5, 'oras5': args.oras5}
        
        ## Retrieve hyperparameters
        config_filepath = Path(f'chaosbench/configs/{args.model_name}.yaml')
        with open(config_filepath, 'r') as config_f:
            hyperparams = yaml.load(config_f, Loader=yaml.FullLoader)

        model_args = hyperparams['model_args']
        data_args = hyperparams['data_args']
        
        ## Load model from checkpoint
        baselines = []
        for version_num in args.version_nums:
            baseline = model.S2SBenchmarkModel(model_args=model_args, data_args=data_args)
            ckpt_filepath = log_dir / f'lightning_logs/version_{version_num}/checkpoints/'
            ckpt_filepath = list(ckpt_filepath.glob('*.ckpt'))[0]
            baseline = baseline.load_from_checkpoint(ckpt_filepath)
            baselines.append(copy.deepcopy(baseline.eval()))
        
        land_vars = baseline.hparams.get('data_args', {}).get('land_vars', [])
        ocean_vars = baseline.hparams.get('data_args', {}).get('ocean_vars', [])

        ## Prepare input/output dataset
        input_dataset = dataset.S2SObsDataset(args.eval_years, config.N_STEPS-1, land_vars=land_vars, ocean_vars=ocean_vars)
        input_dataloader = DataLoader(input_dataset, batch_size=BATCH_SIZE, shuffle=False)

        output_dataset = dataset.S2SObsDataset(
            args.eval_years, config.N_STEPS-1, 
            land_vars=config.LRA5_PARAMS, ocean_vars=config.ORAS5_PARAMS, is_normalized=False
        )
        output_dataloader = DataLoader(output_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
    else:
        raise NotImplementedError('Inference type has yet to be implemented...')
    
    ##################### Initialize criteria ######################
    ####################### Deterministic ##########################
    RMSE = criterion.RMSE()
    Bias = criterion.Bias()
    ACC = criterion.ACC()
    SSIM = criterion.MS_SSIM()
    SpecDiv = criterion.SpectralDiv(percentile=0.9, is_train=False)
    SpecRes = criterion.SpectralRes(percentile=0.9, is_train=False)
    
    ####################### Probabilistic ##########################
    CRPS = criterion.CRPS()
    CRPSS = criterion.CRPSS()
    Spread = criterion.Spread()
    SSR = criterion.SSR()
    ################################################################
    
    
    ######################################
    ####### Evaluation main script #######
    ######################################
    
    
    ## All metric placeholders
    all_rmse, all_bias, all_acc, all_ssim, all_sdiv, all_sres = list(), list(), list(), list(), list(), list()
    all_crps, all_crpss, all_spread, all_ssr = list(), list(), list(), list()
    
    for input_batch, output_batch in tqdm(zip(input_dataloader, output_dataloader), total=len(input_dataloader)):

        _, preds_x, preds_y = input_batch
        timestamps, truth_x, truth_y = output_batch
        
        # Pre-processing (e.g., get day-of-years for climatology-related metrics...)
        doys = utils.get_doys_from_timestep(timestamps)

        assert preds_y.size(1) == truth_y.size(1)
        N_STEPS = preds_y.size(1)
        N_ENSEM = preds_y.size(2) if IS_NWPS else len(baselines)

        ## Step metric placeholders
        step_rmse, step_bias, step_acc, step_ssim, step_sdiv, step_sres = dict(), dict(), dict(), dict(), dict(), dict()
        step_crps, step_crpss, step_spread, step_ssr = dict(), dict(), dict(), dict()

        for step_idx in range(N_STEPS):
            all_param_idx, param_idx = 0, 0

            with torch.no_grad():
                
                ##################### Getting predictions #####################
                if IS_NWPS:
                    preds = preds_y[:, step_idx]
                
                elif IS_AI_MODEL:
                    if step_idx == 0:
                        preds_x = preds_x.unsqueeze(1)
                        preds_x = preds_x.repeat(1, N_ENSEM, 1, 1, 1)
                    
                    # Collect preds across members
                    preds = []
                    for b_id, baseline in enumerate(baselines):
                        preds.append(
                            baseline(preds_x[:,b_id].to(config.device))
                        )
                    preds = torch.stack(preds, dim=1)
                ###############################################################
                
                ####################### Getting targets #######################
                targs = truth_y[:, step_idx]
                targs = targs.unsqueeze(1) # dim=ensemble_num
                targs = targs.repeat(1, N_ENSEM, 1, 1, 1)
                ###############################################################


                ## Extract metric for each param/level
                for i, (param_class, params) in enumerate(ALL_PARAM_LIST.items()):
                    for j, param in enumerate(params):
                        
                        ### Some param/level pairs are not available 
                        param_exist = param in PARAM_LIST[param_class]

                        ## Handling predictions
                        unique_preds = preds[:, :, param_idx] if param_exist else torch.full((BATCH_SIZE, N_ENSEM, 121,240), torch.nan)
                        unique_preds = utils.denormalize(unique_preds, param, param_class) if IS_AI_MODEL else unique_preds
                        unique_preds = unique_preds.double().to(config.device)

                        ## Handling labels
                        unique_labels = targs[:, :, all_param_idx]
                        unique_labels = unique_labels.double().to(config.device)

                        ########################################### Criterion 1: RMSE ##############################################
                        error = RMSE(unique_preds.mean(axis=1), unique_labels.mean(axis=1)).cpu().numpy()

                        ######################################## Criterion 2: Bias #################################################
                        bias = Bias(unique_preds.mean(axis=1), unique_labels.mean(axis=1)).cpu().numpy()

                        ######################################## Criterion 3: ACC ##################################################
                        acc = ACC(unique_preds.mean(axis=1), unique_labels.mean(axis=1), doys[:,step_idx], param, param_class)
                        acc = acc.cpu().numpy()

                        ######################################## Criterion 4: SSIM #################################################
                        ssim = SSIM(unique_preds.mean(axis=1), unique_labels.mean(axis=1)).cpu().numpy()

                        ###################################### Criterion 5: SpecDiv ################################################
                        sdiv = SpecDiv(unique_preds.mean(axis=1), unique_labels.mean(axis=1)).cpu().numpy()

                        ###################################### Criterion 6: SpecRes ################################################
                        sres = SpecRes(unique_preds.mean(axis=1), unique_labels.mean(axis=1)).cpu().numpy()
                        
                        ######################################## Criterion 7a: CRPS ################################################
                        crps = CRPS(unique_preds, unique_labels.mean(axis=1)).cpu().numpy()
                        
                        ########################################## Criterion 7b: CRPSS #############################################
                        crpss = CRPSS(unique_preds, unique_labels.mean(axis=1), doys[:,step_idx], param, param_class).cpu().numpy()
                        
                        ####################################### Criterion 8: Spread ################################################
                        spread = Spread(unique_preds, unique_labels.mean(axis=1)).cpu().numpy()
                        
                        ######################################## Criterion 9: SSR ##################################################
                        ssr = SSR(unique_preds, unique_labels.mean(axis=1)).cpu().numpy()
                        
                        try:
                            step_rmse[param].extend([error])
                            step_bias[param].extend([bias])
                            step_acc[param].extend([acc])
                            step_ssim[param].extend([ssim])
                            step_sdiv[param].extend([sdiv])
                            step_sres[param].extend([sres])
                            step_crps[param].extend([crps])
                            step_crpss[param].extend([crpss])
                            step_spread[param].extend([spread])
                            step_ssr[param].extend([ssr])

                        except:
                            step_rmse[param] = [error]
                            step_bias[param] = [bias]
                            step_acc[param] = [acc]
                            step_ssim[param] = [ssim]
                            step_sdiv[param] = [sdiv]
                            step_sres[param] = [sres]
                            step_crps[param] = [crps]
                            step_crpss[param] = [crpss]
                            step_spread[param] = [spread]
                            step_ssr[param] = [ssr]
                            
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
        all_crps.append(step_crps)
        all_crpss.append(step_crpss)
        all_spread.append(step_spread)
        all_ssr.append(step_ssr)
    
    ## Combine metrics across batch
    merged_rmse, merged_bias, merged_acc, merged_ssim, merged_sdiv, merged_sres, \
    merged_crps, merged_crpss, merged_spread, merged_ssr = dd(list), dd(list), dd(list), dd(list), dd(list), dd(list), dd(list), dd(list), dd(list), dd(list)
    
    for d_rmse, d_bias, d_acc, d_ssim, d_sdiv, d_sres, \
        d_crps, d_crpss, d_spread, d_ssr in zip(all_rmse, all_bias, all_acc, all_ssim, all_sdiv, all_sres, all_crps, all_crpss, all_spread, all_ssr):
        
        for (rmse_k, rmse_v), (bias_k, bias_v), (acc_k, acc_v), (ssim_k, ssim_v), (sdiv_k, sdiv_v), (sres_k, sres_v), \
            (crps_k, crps_v), (crpss_k, crpss_v), (spread_k, spread_v), (ssr_k, ssr_v) in zip(d_rmse.items(),
                                                                                              d_bias.items(),
                                                                                              d_acc.items(),
                                                                                              d_ssim.items(),
                                                                                              d_sdiv.items(),
                                                                                              d_sres.items(),
                                                                                              d_crps.items(),
                                                                                              d_crpss.items(),
                                                                                              d_spread.items(),
                                                                                              d_ssr.items()):
            
            merged_rmse[rmse_k].append(rmse_v)
            merged_bias[bias_k].append(bias_v)
            merged_acc[acc_k].append(acc_v)
            merged_ssim[ssim_k].append(ssim_v)
            merged_sdiv[sdiv_k].append(sdiv_v)
            merged_sres[sres_k].append(sres_v)
            merged_crps[crps_k].append(crps_v)
            merged_crpss[crpss_k].append(crpss_v)
            merged_spread[spread_k].append(spread_v)
            merged_ssr[ssr_k].append(ssr_v)

    ## Compute the mean metrics over valid evaluation time horizon (for each timestep) along batch
    merged_rmse, merged_bias, merged_acc, merged_ssim, merged_sdiv, merged_sres, \
    merged_crps, merged_crpss, merged_spread, merged_ssr = dict(merged_rmse), dict(merged_bias), dict(merged_acc), dict(merged_ssim), dict(merged_sdiv), dict(merged_sres), dict(merged_crps), dict(merged_crpss), dict(merged_spread), dict(merged_ssr)
    
    for (rmse_k, rmse_v), (bias_k, bias_v), (acc_k, acc_v), (ssim_k, ssim_v), (sdiv_k, sdiv_v), (sres_k, sres_v), \
        (crps_k, crps_v), (crpss_k, crpss_v), (spread_k, spread_v), (ssr_k, ssr_v) in zip(merged_rmse.items(), 
                                                                                          merged_bias.items(),
                                                                                          merged_acc.items(),
                                                                                          merged_ssim.items(),
                                                                                          merged_sdiv.items(),
                                                                                          merged_sres.items(),
                                                                                          merged_crps.items(),
                                                                                          merged_crpss.items(),
                                                                                          merged_spread.items(),
                                                                                          merged_ssr.items()):
        
        merged_rmse[rmse_k] = np.array(merged_rmse[rmse_k]).mean(axis=0)
        merged_bias[bias_k] = np.array(merged_bias[bias_k]).mean(axis=0)
        merged_acc[acc_k] = np.array(merged_acc[acc_k]).mean(axis=0)
        merged_ssim[ssim_k] = np.array(merged_ssim[ssim_k]).mean(axis=0)
        merged_sdiv[sdiv_k] = np.array(merged_sdiv[sdiv_k]).mean(axis=0)
        merged_sres[sres_k] = np.array(merged_sres[sres_k]).mean(axis=0)
        merged_crps[crps_k] = np.array(merged_crps[crps_k]).mean(axis=0)
        merged_crpss[crpss_k] = np.array(merged_crpss[crpss_k]).mean(axis=0)
        merged_spread[spread_k] = np.array(merged_spread[spread_k]).mean(axis=0)
        merged_ssr[ssr_k] = np.array(merged_ssr[ssr_k]).mean(axis=0)
        
    ## Save metrics
    eval_dir = log_dir / 'eval'
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame(merged_rmse).to_csv(eval_dir / f'rmse_{args.model_name}.csv', index=False)
    pd.DataFrame(merged_bias).to_csv(eval_dir / f'bias_{args.model_name}.csv', index=False)
    pd.DataFrame(merged_acc).to_csv(eval_dir / f'acc_{args.model_name}.csv', index=False)
    pd.DataFrame(merged_ssim).to_csv(eval_dir / f'ssim_{args.model_name}.csv', index=False)
    pd.DataFrame(merged_sdiv).to_csv(eval_dir / f'sdiv_{args.model_name}.csv', index=False)
    pd.DataFrame(merged_sres).to_csv(eval_dir / f'sres_{args.model_name}.csv', index=False)
    pd.DataFrame(merged_crps).to_csv(eval_dir / f'crps_{args.model_name}.csv', index=False)
    pd.DataFrame(merged_crpss).to_csv(eval_dir / f'crpss_{args.model_name}.csv', index=False)
    pd.DataFrame(merged_spread).to_csv(eval_dir / f'spread_{args.model_name}.csv', index=False)
    pd.DataFrame(merged_ssr).to_csv(eval_dir / f'ssr_{args.model_name}.csv', index=False)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help='Name of the model specified in your config file, including one of [ecmwf, cma, ukmo, ncep, persistence]')
    parser.add_argument('--eval_years', nargs='+', help='Provide the evaluation years')
    parser.add_argument('--version_nums', nargs='+', help='Version numbers of the model from multiple checkpoints')
    parser.add_argument('--lra5', nargs='+', type=str, default=[], help='List of LRA5 variables to be evaluated')
    parser.add_argument('--oras5', nargs='+', type=str, default=[], help='List of ORAS5 variables to be evaluated')    
    
    args = parser.parse_args()
    main(args)
