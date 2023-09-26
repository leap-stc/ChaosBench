import os
import argparse
from pathlib import Path
import yaml

import torch
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
pl.seed_everything(42)

from chaosbench.models import model


def main(args):
    """
    Training script given .yaml config
    Example usage:
        1) `python train.py --config_filepath chaosbench/configs/mlp_s2s.yaml`
    """
    
    # Retrieve hyperparameters
    with open(args.config_filepath, 'r') as config_filepath:
        hyperparams = yaml.load(config_filepath, Loader=yaml.FullLoader)
        
    model_args = hyperparams['model_args']
    data_args = hyperparams['data_args']
        
    # Initialize model
    baseline = model.S2SBenchmarkModel(model_args=model_args, data_args=data_args)
    baseline.setup()
    
    # Initialize training
    log_dir = Path('logs') / model_args['model_name']
    
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min')

    trainer = pl.Trainer(
        devices=-1,
        accelerator='gpu',
        strategy='auto',
        max_epochs=model_args['epochs'],
        logger=tb_logger,
        callbacks=[checkpoint_callback]
     )

    trainer.fit(baseline)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filepath', help='Provide the filepath string to the model config...')
    
    args = parser.parse_args()
    main(args)
