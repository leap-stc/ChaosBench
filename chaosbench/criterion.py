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


class MS_SSIM(nn.Module):
    """
    Compute Multi-Scale Structural SIMilarity(MS-SSIM) index
    """
    
    def __init__(
        self,
        data_range=255,
        size_average=True,
        kernel_size=11,
        sigma=1.5,
        weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
        k1=0.01,
        k2=0.03
    ):
        """
        Args:
            data_range: max-min, usually use 1 or 255
            kernel_size: size of the Gaussian kernel
            sigma: standard deviation of the Gaussian kernel
            
        """
        super(MS_SSIM, self).__init__()
        self.data_range = data_range
        self.size_average = size_average
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.weights = weights
        self.k1 = k1
        self.k2 = k2

    def rescale(self, data):
        """
        (B, H, W) -> (B,1,H,W) and rescale to (0,255) for each sample
        """
        # Add the additional axis to the data
        data_reshaped = data.unsqueeze(1)
        data_rescaled = torch.zeros_like(data_reshaped)
        
        for i in range(len(data)):
            min_val = data_reshaped[i].min()
            max_val = data_reshaped[i].max()
            data_rescaled[i] = self.data_range * (data_reshaped[i] - min_val) / (max_val - min_val)
        
        return data_rescaled
        
        
    def gaussian_1d(self):
        """
        1-d Gaussian filter
        """
        coords = torch.arange(self.kernel_size, dtype=torch.float)
        coords -= self.kernel_size // 2
        
        g = torch.exp(-(coords ** 2)/(2*self.sigma**2))
        g /= g.sum()
        
        return g.unsqueeze(0).unsqueeze(0)
    
    
    def gaussian_filter(self, data, gaussian_kernel):
        """
        Gaussian filtering
        """
        conv = nn.functional.conv2d
        C = data.shape[1]
        out = data
        for i, s in enumerate(data.shape[2:]):
            if s>= gaussian_kernel.shape[-1]:
                out = conv(out, weight=gaussian_kernel.transpose(2+i, -1), stride=1, padding=0, groups=C)
                
        return out
            
        
    def ssim(self,
             X,
             Y,
             gaussian_kernel,
    ):
        C1 = (self.k1 * self.data_range) ** 2
        C2 = (self.k2 * self.data_range) ** 2
        
        compensation = 1.0
        
        gaussian_kernel = gaussian_kernel.to(X.device, dtype=X.dtype)
        
        mu1 = self.gaussian_filter(X, gaussian_kernel)
        mu2 = self.gaussian_filter(Y, gaussian_kernel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = compensation * (self.gaussian_filter(X * X, gaussian_kernel) - mu1_sq)
        sigma2_sq = compensation * (self.gaussian_filter(Y * Y, gaussian_kernel) - mu2_sq)
        sigma12 = compensation * (self.gaussian_filter(X * Y, gaussian_kernel) - mu1_mu2)

        cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
        ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

        ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
        cs = torch.flatten(cs_map, 2).mean(-1)
        return ssim_per_channel, cs
    
    def ms_ssim(
        self,
        predictions,
        targets,
        size_average=True
    ):
        
        """
        predictions: a batch of a specific predicted physical variable at a specific level (B,H,W)
        targets: a batch of a specific target physical variable at a specific level (B,H,W)
        """
        predictions = self.rescale(predictions).squeeze(dim=-1).squeeze(dim=-1)
        targets = self.rescale(targets).squeeze(dim=-1).squeeze(dim=-1)
        
        avg_pool = nn.functional.avg_pool2d
        
        window = self.gaussian_1d()
        window = window.repeat([predictions.shape[1]] + [1] * ( len(predictions.shape)-1))
        
        weights_tensor = predictions.new_tensor(self.weights)
        
        levels = len(self.weights)
        mcs = []
        for i in range(levels):
            ssim_per_channel, cs = self.ssim(predictions, targets, window)
            
            if i < levels - 1:
                mcs.append(torch.relu(cs))
                padding = [s%2 for s in predictions.shape[2:]]
                predictions = avg_pool(predictions, kernel_size=2, padding=padding)
                targets = avg_pool(targets, kernel_size=2, padding=padding)
        
        ssim_per_channel = torch.relu(ssim_per_channel)  
        mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  
        ms_ssim_val = torch.prod(mcs_and_ssim ** weights_tensor.view(-1, 1, 1), dim=0)

        if size_average:
        
            return ms_ssim_val.mean()
        else:
        
            return ms_ssim_val.mean(1)

                
    def forward(self, predictions, targets):
        return self.ms_ssim(
            predictions,
            targets,
        )
