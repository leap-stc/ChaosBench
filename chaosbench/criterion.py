import torch
import torch.nn as nn
import xarray as xr
from pathlib import Path

from chaosbench import config

def get_adjusting_weights():
    latitudes = torch.arange(90, -91.5, -1.5)
    latitudes_rad = torch.deg2rad(latitudes)
    weights = torch.cos(latitudes_rad)
    
    return weights[None, :, None].to(config.device)

class RMSE(nn.Module):
    """
    Compute root mean squared error (RMSE)
    """
    
    def __init__(self,
                 lat_adjusted=True):
        super(RMSE, self).__init__()
        
        self.lat_adjusted = lat_adjusted
        self.weights = get_adjusting_weights() if lat_adjusted else None

    def forward(self, predictions, targets):
        
        # Calculate the squared differences between predictions and targets
        squared_diff = (predictions - targets) ** 2
        
        # Adjust by latitude
        if self.lat_adjusted:
            squared_diff = self.weights.size(1) * (self.weights / torch.sum(self.weights)) * squared_diff 
            
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
    
    def __init__(self,
                lat_adjusted=True):
        super(Bias, self).__init__()
        
        self.lat_adjusted = lat_adjusted
        self.weights = get_adjusting_weights() if lat_adjusted else None
        
    def forward(self, predictions, targets):
        
        # Calculate difference between predictions and targets
        bias = predictions - targets
        
        if self.lat_adjusted:
            bias = self.weights.size(1) * (self.weights / torch.sum(self.weights)) * bias 
        
        # Calculate the mean bias
        mean_bias = torch.nanmean(bias)
        
        return mean_bias
    
class MAE(nn.Module):
    """Compute mean absolute error
    """
    
    def __init__(self,
                 lat_adjusted=True):
        
        super(MAE, self).__init__()
        
        self.lat_adjusted = lat_adjusted
        self.weights = get_adjusting_weights() if lat_adjusted else None
        
    def forward(self, predictions, targets):
        
        if self.lat_adjusted:
            predictions = self.weights.size(1) * (self.weights / torch.sum(self.weights)) * predictions 
            targets = self.weights.size(1) * (self.weights / torch.sum(self.weights)) * targets 
        
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
    
    def __init__(self,
                 lat_adjusted=True):
         
        super(R2, self).__init__()
        
        self.lat_adjusted = lat_adjusted
        self.weights = get_adjusting_weights() if lat_adjusted else None
        
    def forward(self, predictions, targets):
         
        if self.lat_adjusted:
            predictions = self.weights.size(1) * (self.weights / torch.sum(self.weights)) * predictions
            targets = self.weights.size(1) * (self.weights / torch.sum(self.weights)) * targets
        
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
    def __init__(self,
                 lat_adjusted=True):
        
        super(ACC, self).__init__()
        
        self.lat_adjusted = lat_adjusted
        self.weights = get_adjusting_weights() if lat_adjusted else None
        
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
         
         
        if self.lat_adjusted:
            anomalies_targets = self.weights.size(1) * (self.weights / torch.sum(self.weights)) * anomalies_targets
            anomalies_predictions = self.weights.size(1) * (self.weights / torch.sum(self.weights)) * anomalies_predictions

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