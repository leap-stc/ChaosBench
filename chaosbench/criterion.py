import torch
import torch.nn as nn
import xarray as xr
from pathlib import Path

from chaosbench import config

class RMSE(nn.Module):
    """
    Compute root mean squared error (RMSE)
    """
    
    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, predictions, targets):
        
        # Calculate the squared differences between predictions and targets
        squared_diff = (predictions - targets) ** 2
        
        # Calculate the mean squared error
        mean_squared_error = torch.nanmean(squared_diff)
        
        # Take the square root to get the RMSE
        rmse = torch.sqrt(mean_squared_error)
        
        return rmse
    
class MSE(nn.Module):
    """
    Compute mean squared error (MSE)
    """
    
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, predictions, targets):
        
        # Calculate the squared differences between predictions and targets
        squared_diff = (predictions - targets) ** 2
        
        # Calculate the mean squared error
        mean_squared_error = torch.nanmean(squared_diff)
        
        return mean_squared_error
    
    
class Bias(nn.Module):
    """Compute bias (predictions - targets)
    """
    
    def __init__(self):
        super(Bias, self).__init__()
        
    def forward(self, predictions, targets):
        
        # Calculate difference between predictions and targets
        bias = predictions - targets
        
        # Calculate the mean bias
        mean_bias = torch.nanmean(bias)
        
        return mean_bias
    
class MAE(nn.Module):
    """Compute mean absolute error
    """
    
    def __init__(self):
        super(MAE, self).__init__()
        
    def forward(self, predictions, targets):
        
        # Calculate difference
        absolute_diff = torch.abs(predictions - targets)
        
        # Calculate the mean absolute difference
        mean_absolute_error = torch.nanmean(absolute_diff)
        
        return mean_absolute_error
    
class R2(nn.Module):
    """
    Compute R^2 = 1 - (RSS/TSS)
    where, RSS = sum of square residual; TSS = total sum of squares
    """
    
    def __init__(self):
        super(R2, self).__init__()
        
    def forward(self, predictions, targets):
        
        # Compute RSS and TSS
        mean_targets = torch.nanmean(targets)   
        rss = torch.nansum((targets - predictions) ** 2)     
        tss = torch.nansum((targets - mean_targets) ** 2)
        
        # Compute r2
        r2 = 1 - (rss / tss)
        
        return r2
    
class ACC(nn.Module):
    """
    Compute anomaly correlation coefficient (ACC) given climatology
    """
    def __init__(self):
        super(ACC, self).__init__()
        
        # Retrieve climatology
        self.normalization_file = Path(config.DATA_DIR) / 's2s' / 'climatology' / 'climatology_era5.zarr'
        self.normalization = xr.open_dataset(self.normalization_file, engine='zarr')
        self.normalization_mean = self.normalization['mean'].values
        
    def forward(self, predictions, targets, param, level):
        
        # Retrieve mean climatology
        climatology = self.normalization_mean[config.PARAMS.index(param), config.PRESSURE_LEVELS.index(level)]
        
        # Compute anomalies
        anomalies_targets = targets - climatology
        anomalies_predictions = predictions - climatology

        # Compute ACC
        numerator = torch.nansum(anomalies_targets * anomalies_predictions)
        denominator = torch.sqrt(torch.nansum(anomalies_targets ** 2) * torch.nansum(anomalies_predictions ** 2))

        acc = numerator / denominator
        
        return acc
    
class KL_MSE(nn.Module):
    """
    Compute mean squared error (MSE) and KL-divergence
    """
    
    def __init__(self):
        super(KL_MSE, self).__init__()

    def forward(self, predictions, targets):
        
        predictions, mu, logvar = predictions
        
        # Calculate the squared differences between predictions and targets
        squared_diff = (predictions - targets) ** 2
        
        # Calculate the mean squared error
        mean_squared_error = torch.nanmean(squared_diff)
        
        # Compute KL-divergence
        kld_loss = -0.5 * torch.nansum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return mean_squared_error + kld_loss