import os
import torch
torch.autograd.set_detect_anomaly(True)
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import lightning.pytorch as pl
import yaml

from chaosbench.models import mlp, cnn, ae, fno, vit
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
        land_vars = self.data_args.get('land_vars', [])
        ocean_vars = self.data_args.get('ocean_vars', [])
        input_size = self.model_args['input_size'] + len(land_vars) + len(ocean_vars)
        output_size = self.model_args['input_size'] + len(land_vars) + len(ocean_vars)
        
        if 'mlp' in self.model_args['model_name']:
            self.model = mlp.MLP(input_size = input_size,
                                 hidden_sizes = self.model_args['hidden_sizes'], 
                                 output_size = output_size)
            
        elif 'unet' in self.model_args['model_name']:
            self.model = cnn.UNet(input_size = input_size,
                                   output_size = output_size)
            
        elif 'resnet' in self.model_args['model_name']:
            self.model = cnn.ResNet(input_size = input_size,
                                   output_size = output_size)
            
        elif 'vae' in self.model_args['model_name']:
            self.model = ae.VAE(input_size = input_size,
                                 output_size = output_size,
                                 latent_size = self.model_args['latent_size'])
            
        elif 'ed' in self.model_args['model_name']:
            self.model = ae.EncoderDecoder(input_size = input_size,
                                           output_size = output_size)
            
        elif 'fno' in self.model_args['model_name']:
            self.model = fno.FNO2d(input_size = input_size,
                                   modes1 = self.model_args['modes1'], 
                                   modes2 = self.model_args['modes2'], 
                                   width = self.model_args['width'], 
                                   initial_step = self.model_args['initial_step'])
            
        elif 'segformer' in self.model_args['model_name']:
            self.model = vit.Segformer(input_size = input_size)
            
        
        ##################################
        # INITIALIZE YOUR OWN MODEL HERE #
        ##################################
        
        self.loss = self.init_loss_fn()
            
    def init_loss_fn(self):
        loss = criterion.MSE()
        return loss
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        timestamp, x, y = batch
        
        ################## Iterative loss ##################
        n_steps = y.size(1)
        loss = 0
        
        for step_idx in range(n_steps):
            preds = self(x)
            
            # Optimize for headline variables
            if self.model_args['only_headline']:
                headline_idx = [
                    config.PARAMS.index(headline_var.split('-')[0]) * len(config.PRESSURE_LEVELS)
                    + config.PRESSURE_LEVELS.index(int(headline_var.split('-')[1])) for headline_var in config.HEADLINE_VARS
                ]
                
                loss += self.loss(
                    preds.view(preds.size(0), -1, preds.size(-2), preds.size(-1))[:, headline_idx],
                    y[:, step_idx].view(y.size(0), -1, y.size(-2), y.size(-1))[:, headline_idx]
                )
            
            # Otherwise, for all variables
            else:
                loss += self.loss(preds[:, :self.model_args['output_size']], y[:, step_idx, :self.model_args['output_size']])
            
            x = preds
            
        loss = loss / n_steps
        ####################################################
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        timestamp, x, y = batch
        
        ################## Iterative loss ##################
        n_steps = y.size(1)
        loss = 0
        
        for step_idx in range(n_steps):
            preds = self(x)
            
            # Optimize for headline variables
            if self.model_args['only_headline']:
                headline_idx = [
                    config.PARAMS.index(headline_var.split('-')[0]) * len(config.PRESSURE_LEVELS)
                    + config.PRESSURE_LEVELS.index(int(headline_var.split('-')[1])) for headline_var in config.HEADLINE_VARS
                ]
                
                loss += self.loss(
                    preds.view(preds.size(0), -1, preds.size(-2), preds.size(-1))[:, headline_idx],
                    y[:, step_idx].view(y.size(0), -1, y.size(-2), y.size(-1))[:, headline_idx]
                )
                
            # Otherwise, for all variables
            else:
                loss += self.loss(preds[:, :self.model_args['output_size']], y[:, step_idx, :self.model_args['output_size']])
            
            x = preds
            
        loss = loss / n_steps
        ####################################################
        
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.model_args['learning_rate'])
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': CosineAnnealingLR(optimizer, T_max=self.model_args['t_max'], eta_min=self.model_args['learning_rate'] / 10),
                'interval': 'epoch',
            }
        }

    def setup(self, stage=None):
        self.train_dataset = dataset.S2SObsDataset(years=self.data_args['train_years'], 
                                                   n_step=self.data_args['n_step'],
                                                   lead_time=self.data_args['lead_time'],
                                                   land_vars=self.data_args['land_vars'],
                                                   ocean_vars=self.data_args['ocean_vars']
                                                  )
        self.val_dataset = dataset.S2SObsDataset(years=self.data_args['train_years'], 
                                                 n_step=self.data_args['n_step'],
                                                 lead_time=self.data_args['lead_time'],
                                                 land_vars=self.data_args['land_vars'],
                                                 ocean_vars=self.data_args['ocean_vars']
                                                )
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          num_workers=self.model_args['num_workers'], 
                          batch_size=self.data_args['batch_size'], shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          num_workers=self.model_args['num_workers'], 
                          batch_size=self.data_args['batch_size'])
