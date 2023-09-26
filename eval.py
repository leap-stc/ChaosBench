import os
import argparse
import yaml
import numpy as np
import pandas as pd
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

from chaosbench import dataset, config, utils, criterion
from chaosbench.models import mlp, model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    """
    Evaluation script given .yaml config and trained model checkpoint
    
    Example usage:
        (Persistence)              1) `python eval.py --model_name persistence --eval_years 2021 2022`
        (Physical models)          3) `python eval.py --model_name ecmwf --eval_years 2021 2022`
        (Data-driven models)       4) `python eval.py --model_name mlp_s2s --eval_years 2021 2022`
    
    """
    print(f'Evaluating ERA5 observations against {args.model_name}...')
    
    #########################################
    ####### Evaluation initialization #######
    #########################################
    
    IS_AI_MODEL, IS_PERSISTENCE = False, False
    BATCH_SIZE = 32
    
    ## Prepare directory to load model / store metrics
    log_dir = Path('logs') / args.model_name
    eval_dir = log_dir / 'eval'
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    ## Case 1: Persistence
    if args.model_name == 'persistence':
        IS_PERSISTENCE = True
        
        input_dataset = dataset.S2SObsDataset(years=args.eval_years, is_val=True)
        input_dataloader = DataLoader(input_dataset, batch_size=BATCH_SIZE, shuffle=False)
        output_dataset = dataset.S2SObsDataset(years=args.eval_years, is_val=True)
        output_dataloader = DataLoader(output_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
    
    ## Case 2: Physical models, one of [ecmwf, cma, ukmo, ncep]
    elif args.model_name in list(config.S2S_CENTERS.keys()):
        input_dataset = dataset.S2SEvalDataset(s2s_name=args.model_name, years=args.eval_years)
        input_dataloader = DataLoader(input_dataset, batch_size=BATCH_SIZE, shuffle=False)
        output_dataset = dataset.S2SObsDataset(years=args.eval_years, is_val=True)
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
        ckpt_filepath = log_dir / 'lightning_logs/version_0/checkpoints/'
        ckpt_filepath = list(ckpt_filepath.glob('*.ckpt'))[0]
        baseline = baseline.load_from_checkpoint(ckpt_filepath)
        
        ## Prepare input/output dataset
        input_dataset = dataset.S2SObsDataset(years=args.eval_years, is_val=True)
        input_dataloader = DataLoader(input_dataset, batch_size=data_args['batch_size'], shuffle=False)
        output_dataset = dataset.S2SObsDataset(years=args.eval_years, is_val=True)
        output_dataloader = DataLoader(output_dataset, batch_size=data_args['batch_size'], shuffle=False)
        
    
    ##### Initialize criteria #####
    RMSE = criterion.RMSE()
    Bias = criterion.Bias()
    MAE = criterion.MAE()
    R2 = criterion.R2()
    ACC = criterion.ACC()
    ###############################
    
    
    ######################################
    ####### Evaluation main script #######
    ######################################
    
    ## All metric placeholders
    all_rmse, all_bias, all_mae, all_r2, all_acc = list(), list(), list(), list(), list()

    for input_batch, output_batch in tqdm(zip(input_dataloader, output_dataloader), total=len(input_dataloader)):
        
        _, test_x = input_batch # of shape (batch, step, param, level, lat, lon)
        _, test_y = output_batch # of shape (batch, step, param, level, lat, lon)

        ## Removing days where there are no forecasts
        mask = (torch.isnan(test_x).sum(dim=(1,2,3,4,5)) != torch.prod(torch.tensor(test_x.shape[1:])))
        test_x = test_x[mask]
        test_y = test_y[mask]
        
        assert config.N_STEPS == test_x.size(1)

        test_x = torch.nan_to_num(test_x, nan=0.0)
        curr_x = test_x[:, 0].to(device)

        ## Step metric placeholders
        step_rmse, step_bias, step_mae, step_r2, step_acc = dict(), dict(), dict(), dict(), dict()

        for step_idx in range(config.N_STEPS - 1):

            with torch.no_grad():
                curr_y = test_y[:, step_idx + 1].to(device)
                
                
                ############## Getting predictions ##############
                
                if IS_AI_MODEL:
                    preds = baseline(curr_x)
                
                elif IS_PERSISTENCE:
                    preds = test_x[:, 0].to(device)
                
                else:
                    preds = test_x[:, step_idx + 1].to(device)
                
                #################################################
                

                ## Extract metric for each param/level
                for param_idx, param in enumerate(config.PARAMS):
                    for level_idx, level in enumerate(config.PRESSURE_LEVELS):

                        ## Denormalize
                        preds[:, param_idx, level_idx] = utils.denormalize(
                            preds[:, param_idx, level_idx], param, level, args.model_name
                        )

                        curr_y[:, param_idx, level_idx] = utils.denormalize(
                            curr_y[:, param_idx, level_idx], param, level, 'era5'
                        )

                        ################################## Criterion 1: RMSE #####################################
                        error = RMSE(preds[:, param_idx, level_idx], curr_y[:, param_idx, level_idx]).cpu().numpy()

                        ################################## Criterion 2: Bias #####################################
                        bias = Bias(preds[:, param_idx, level_idx], curr_y[:, param_idx, level_idx]).cpu().numpy()

                        ################################## Criterion 3: MAE ######################################
                        mae = MAE(preds[:, param_idx, level_idx], curr_y[:, param_idx, level_idx]).cpu().numpy()

                        ################################## Criterion 4: R2 #######################################
                        r2 = R2(preds[:, param_idx, level_idx], curr_y[:, param_idx, level_idx]).cpu().numpy()
                        
                        ################################## Criterion 4: ACC ######################################
                        acc = ACC(preds[:, param_idx, level_idx], curr_y[:, param_idx, level_idx], param, level).cpu().numpy()


                        try:
                            step_rmse[f'{param}-{level}'].extend([error])
                            step_bias[f'{param}-{level}'].extend([bias])
                            step_mae[f'{param}-{level}'].extend([mae])
                            step_r2[f'{param}-{level}'].extend([r2])
                            step_acc[f'{param}-{level}'].extend([acc])

                        except:
                            step_rmse[f'{param}-{level}'] = [error]
                            step_bias[f'{param}-{level}'] = [bias]
                            step_mae[f'{param}-{level}'] = [mae]
                            step_r2[f'{param}-{level}'] = [r2]
                            step_acc[f'{param}-{level}'] = [acc]

                ## Make next-step input as the current prediction (used for AI models)
                curr_x = preds

            ## Cleaning up to release memory
            curr_y, preds = curr_y.cpu().detach(), preds.cpu().detach()
            del curr_y, preds
                    
        test_x, test_y = test_x.cpu().detach(), test_y.cpu().detach()
        del test_x, test_y
        gc.collect()
        
        all_rmse.append(step_rmse)
        all_bias.append(step_bias)
        all_mae.append(step_mae)
        all_r2.append(step_r2)
        all_acc.append(step_acc)
    
    ## Combine metrics
    merged_rmse, merged_bias, merged_mae, merged_r2, merged_acc = dd(list), dd(list), dd(list), dd(list), dd(list)
    
    for d_rmse, d_bias, d_mae, d_r2, d_acc in zip(all_rmse, all_bias, all_mae, all_r2, all_acc):
        for (rmse_k, rmse_v), (bias_k, bias_v), (mae_k, mae_v), (r2_k, r2_v), (acc_k, acc_v) in zip(d_rmse.items(), 
                                                                                                    d_bias.items(),
                                                                                                    d_mae.items(),
                                                                                                    d_r2.items(),
                                                                                                    d_acc.items()):
            merged_rmse[rmse_k].append(rmse_v)
            merged_bias[bias_k].append(bias_v)
            merged_mae[mae_k].append(mae_v)
            merged_r2[r2_k].append(r2_v)
            merged_acc[acc_k].append(acc_v)

    ## Compute the mean metrics over valid evaluation time horizon (for each timestep)
    merged_rmse, merged_bias, merged_mae, merged_r2, merged_acc = dict(merged_rmse), dict(merged_bias), dict(merged_mae), dict(merged_r2), dict(merged_acc)
    for (rmse_k, rmse_v), (bias_k, bias_v), (mae_k, mae_v), (r2_k, r2_v), (acc_k, acc_v) in zip(merged_rmse.items(), 
                                                                                                merged_bias.items(),
                                                                                                merged_mae.items(),
                                                                                                merged_r2.items(),
                                                                                                merged_acc.items()):
        merged_rmse[rmse_k] = np.array(merged_rmse[rmse_k]).mean(axis=0)
        merged_bias[bias_k] = np.array(merged_bias[bias_k]).mean(axis=0)
        merged_mae[mae_k] = np.array(merged_mae[mae_k]).mean(axis=0)
        merged_r2[r2_k] = np.array(merged_r2[r2_k]).mean(axis=0)
        merged_acc[acc_k] = np.array(merged_acc[acc_k]).mean(axis=0)
        
    ## Save metrics
    rmse_df, bias_df, mae_df, r2_df, acc_df = pd.DataFrame(merged_rmse), pd.DataFrame(merged_bias), pd.DataFrame(merged_mae), pd.DataFrame(merged_r2), pd.DataFrame(merged_acc)
    rmse_df.to_csv(eval_dir / f'rmse_{args.model_name}.csv', index=False)
    bias_df.to_csv(eval_dir / f'bias_{args.model_name}.csv', index=False)
    mae_df.to_csv(eval_dir / f'mae_{args.model_name}.csv', index=False)
    r2_df.to_csv(eval_dir / f'r2_{args.model_name}.csv', index=False)
    acc_df.to_csv(eval_dir / f'acc_{args.model_name}.csv', index=False)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help='Name of the model specified in your config file, including one of [ecmwf, cma, ukmo, ncep, persistence]')
    parser.add_argument('--eval_years', nargs='+', help='Provide the evaluation years...')
    
    args = parser.parse_args()
    main(args)
