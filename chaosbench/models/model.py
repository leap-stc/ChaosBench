import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import lightning.pytorch as pl
import yaml

from chaosbench.models import mlp
from chaosbench import dataset, config, utils, criterion

class S2SBenchmarkModel(pl.LightningModule):

    def __init__(
        self, 
        model_args,
        data_args,
        
    ):
        super(S2SBenchmarkModel, self).__init__()
        self.save_hyperparameters()
        
        self.model_args = model_args
        self.data_args = data_args
        
        # Initialize model
        if 'mlp' in self.model_args['model_name']:
            self.model = mlp.MLP(input_size = self.model_args['input_size'],
                                 hidden_sizes = self.model_args['hidden_sizes'], 
                                 output_size = self.model_args['output_size'])
            
        # TODO: more models
            
        # Initialize criteria
        self.mse = criterion.MSE()
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        timestamp, x, y = batch
        preds = self(x)
        loss = self.mse(preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        timestamp, x, y = batch
        preds = self(x)
        loss = self.mse(preds, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.model_args['learning_rate'])
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': CosineAnnealingLR(optimizer, T_max=self.model_args['epochs']),
                'interval': 'epoch',
            }
        }

    def setup(self, stage=None):
        self.train_dataset = dataset.S2SObsDataset(years=self.data_args['train_years'])
        self.val_dataset = dataset.S2SObsDataset(years=self.data_args['train_years'])
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          num_workers=self.model_args['num_workers'], 
                          batch_size=self.data_args['batch_size'], shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          num_workers=self.model_args['num_workers'], 
                          batch_size=self.data_args['batch_size'])
