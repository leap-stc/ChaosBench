import os
import argparse
import yaml
import numpy as np
import pandas as pd
from collections import defaultdict
import gc
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
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
        1) `python eval.py --config_filepath chaosbench/configs/mlp_s2s.yaml --s2s_name ecmwf --eval_years 2021 2022`
    """
    print(f'Evaluating against {args.s2s_name}...')
    
    # Retrieve hyperparameters and setup directories to save metrics...
    with open(args.config_filepath, 'r') as config_filepath:
        hyperparams = yaml.load(config_filepath, Loader=yaml.FullLoader)

    model_args = hyperparams['model_args']
    data_args = hyperparams['data_args']
    
    log_dir = Path('logs') / model_args['model_name']
    eval_dir = log_dir / 'eval'
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    baseline = model.S2SBenchmarkModel(model_args=model_args, data_args=data_args)

    # Load from checkpoint
    ckpt_filepath = log_dir / 'lightning_logs/version_0/checkpoints/'
    ckpt_filepath = list(ckpt_filepath.glob('*.ckpt'))[0]
    baseline = baseline.load_from_checkpoint(ckpt_filepath)
    
    # Prepare evaluation dataset
    test_dataset = dataset.S2SEvalDataset(s2s_name=args.s2s_name, years=args.eval_years)
    test_dataloader = DataLoader(test_dataset, batch_size=data_args['batch_size'], shuffle=False)
    
    # Initialize criteria
    RMSE = criterion.RMSE()
    
    # Evaluation main script
    all_rmse = list()

    for batch in tqdm(test_dataloader):
        timestamp, test_x = batch

        skip_tensor = torch.all(torch.isnan(test_x)) ## Some days do not have data
        step_size = test_x.size(1)

        if not skip_tensor:

            test_x = torch.nan_to_num(test_x, nan=0.0)
            curr_input = test_x[:, 0].to(device)
            
            step_rmse = dict()

            for step_idx in range(step_size):

                with torch.no_grad():
                    test_y = test_x[:, step_idx].to(device)
                    preds = baseline(curr_input)


                    ## Extract error for each param/level
                    for param_idx, param in enumerate(config.PARAMS):
                        for level_idx, level in enumerate(config.PRESSURE_LEVELS):

                            error = utils.denormalize(
                                RMSE(preds[:, param_idx, level_idx], test_y[:, param_idx, level_idx]).cpu().numpy(),
                                param=param, 
                                level=level,
                                dataset_name=args.s2s_name,
                                is_diff=True
                            )

                            try:
                                step_rmse[f'{param}-{level}'].extend([error])

                            except:
                                step_rmse[f'{param}-{level}'] = [error]

                    ## Make next-step input as the current prediction
                    curr_input = preds
                    
                ## Cleaning up to release memory
                test_y, preds = test_y.cpu().detach(), preds.cpu().detach()
                del test_y, preds
                    
            test_x = test_x.cpu().detach()
            del test_x

            all_rmse.append(step_rmse)
            gc.collect()
    
    ## Combine metrics
    merged_rmse = defaultdict(list)
    for d in all_rmse:
        for key, value in d.items():
            merged_rmse[key].append(value)

    ## Compute the mean metrics over valid evaluation time horizon (for each timestep)
    merged_rmse = dict(merged_rmse)
    for key, value in merged_rmse.items():
        merged_rmse[key] = np.array(merged_rmse[key]).mean(axis=0)
        
    ## Save metrics
    rmse_df = pd.DataFrame(merged_rmse)
    rmse_df.to_csv(eval_dir / f'rmse_{args.s2s_name}.csv', index=False)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filepath', help='Provide the filepath string to the model config...')
    parser.add_argument('--s2s_name', help='Provide the s2s center to benchmark against...')
    parser.add_argument('--eval_years', nargs='+', help='Provide the evaluation years...')
    
    args = parser.parse_args()
    main(args)