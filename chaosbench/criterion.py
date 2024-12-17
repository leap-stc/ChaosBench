import os
import torch
import torch.nn as nn
import torch.special as special
import torchist
import xarray as xr
from xskillscore import crps_ensemble, crps_gaussian
from pathlib import Path

from chaosbench import config, utils


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
        
        # Compute only valid values
        valid_mask = ~torch.isnan(predictions) & ~torch.isnan(targets)
        predictions, targets = predictions[valid_mask], targets[valid_mask]
         
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
        self.normalization_file = {
                'era5': os.path.join(config.DATA_DIR, 'climatology', 'climatology_era5.zarr'),
                'lra5': os.path.join(config.DATA_DIR, 'climatology', 'climatology_lra5.zarr'),
                'oras5': os.path.join(config.DATA_DIR, 'climatology', 'climatology_oras5.zarr'),
        }
        
        self.normalization_mean = {
                'era5': xr.open_dataset(self.normalization_file['era5'], engine='zarr')['mean'],
                'lra5': xr.open_dataset(self.normalization_file['lra5'], engine='zarr')['mean'],
                'oras5': xr.open_dataset(self.normalization_file['oras5'], engine='zarr')['mean'],
        }
        
    def forward(self, predictions, targets, doys, param, source):
        
        # Retrieve mean climatology
        climatology = torch.tensor(self.normalization_mean[source].sel(doy=doys, param=param).values).to(config.device)

        # Compute anomalies
        anomalies_targets = targets - climatology
        anomalies_predictions = predictions - climatology
        
        # Compute only valid values
        valid_mask = ~torch.isnan(anomalies_predictions) & ~torch.isnan(anomalies_targets)
        anomalies_predictions, anomalies_targets = anomalies_predictions[valid_mask], anomalies_targets[valid_mask]
        
        if self.lat_adjusted:
            anomalies_targets = self.weights.size(1) * (self.weights / torch.sum(self.weights)) * anomalies_targets
            anomalies_predictions = self.weights.size(1) * (self.weights / torch.sum(self.weights)) * anomalies_predictions

        # Compute ACC
        numerator = torch.nansum(anomalies_targets * anomalies_predictions)
        denominator = torch.sqrt(torch.nansum(anomalies_targets ** 2) * torch.nansum(anomalies_predictions ** 2))

        acc = numerator / (denominator + 1e-10)
        
        return acc
    

class KL_MSE(nn.Module):
    """
    Compute mean squared error (MSE) and KL-divergence, mostly for Variational Autoencoder implementation
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
        g /= torch.nansum(g)
        
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

        ssim_per_channel = torch.nanmean(torch.flatten(ssim_map, 2), dim=-1)
        cs = torch.nanmean(torch.flatten(cs_map, 2), dim=-1)
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
        
        # Handling missing values in predictions
        pred_means = torch.nanmean(predictions, dim=(-2, -1))[:, None, None]
        predictions = torch.where(torch.isnan(predictions), pred_means, predictions)
        
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
            return torch.nanmean(ms_ssim_val)
        
        else:
            return torch.nanmean(ms_ssim_val, dim=1)

                
    def forward(self, predictions, targets):
        
        return self.ms_ssim(
            predictions,
            targets,
        )


class SpectralDiv(nn.Module):
    """
    Compute Spectral divergence given the top-k percentile wavenumber (higher k means higher frequency)
    (1) Validation mode: targeting specific top-k percentile wavenumber (higher k means higher frequency) is permissible
    (2) Training mode: computing metric along the entire wavenumber since some operation e.g., binning is nonautograd-able
    """
    def __init__(
        self,   
        percentile=0.9,
        input_shape=(121,240),
        is_train=True
    ):
        
        super(SpectralDiv, self).__init__()
        
        self.percentile = percentile
        self.is_train = is_train
        
        # Compute the discrete Fourier Transform sample frequencies for a signal of size
        nx, ny = input_shape
        kx = torch.fft.fftfreq(nx) * nx
        ky = torch.fft.fftfreq(ny) * ny
        kx, ky = torch.meshgrid(kx, ky)
        
        self.k = torch.sqrt(kx**2 + ky**2).reshape(-1).to(config.device)
        self.k_low = 0.5
        self.k_upp = torch.max(self.k)
        self.k_nbin = torch.arange(self.k_low, torch.max(self.k), 1).size(0)
        
        # Get percentile index
        self.k_percentile_idx = int(self.k_nbin * self.percentile)
        
    def forward(self, predictions, targets):
        
        predictions = predictions.reshape(predictions.shape[0], -1, predictions.shape[-2], predictions.shape[-1])
        targets = targets.reshape(targets.shape[0], -1, targets.shape[-2], targets.shape[-1])
        
        assert predictions.shape[1] == targets.shape[1]
        nc = predictions.shape[1]
        
        # Handling missing values in predictions
        pred_means = torch.nanmean(predictions, dim=(-2, -1), keepdim=True)
        predictions = torch.where(torch.isnan(predictions), pred_means, predictions)
        
        # Compute along mini-batch
        predictions, targets = torch.nanmean(predictions, dim=0), torch.nanmean(targets, dim=0)
        
        # Transform prediction and targets onto the Fourier space and compute the power
        predictions_power, targets_power = torch.fft.fft2(predictions), torch.fft.fft2(targets)
        predictions_power, targets_power = torch.abs(predictions_power)**2, torch.abs(targets_power)**2

        ## If validation, we can target specific quantiles by binning and sorting
        if not self.is_train:
            predictions_Sk = torchist.histogram(self.k.repeat(nc), self.k_nbin, self.k_low, self.k_upp, weights=predictions_power) \
                            / torchist.histogram(self.k.repeat(nc), self.k_nbin, self.k_low, self.k_upp)

            targets_Sk = torchist.histogram(self.k.repeat(nc), self.k_nbin, self.k_low, self.k_upp, weights=targets_power) \
                        / torchist.histogram(self.k.repeat(nc), self.k_nbin, self.k_low, self.k_upp)
            
            # Extract top-k percentile wavenumber and its corresponding power spectrum
            predictions_Sk = predictions_Sk[self.k_percentile_idx:]
            targets_Sk = targets_Sk[self.k_percentile_idx:]
            
            # Normalize as pdf along ordered k
            predictions_Sk = predictions_Sk / torch.nansum(predictions_Sk)
            targets_Sk = targets_Sk / torch.nansum(targets_Sk)
        
        ## If training, compute the entire power spectrum 
        ## NOTE: targeting specific quantiles is yet to be implemented in autograd (i.e., binning operation)
        else:
            predictions_Sk, targets_Sk = predictions_power, targets_power
        
            # Normalize as pdf of each channel dimension
            predictions_Sk = predictions_Sk / torch.nansum(predictions_Sk, dim=(-2, -1), keepdim=True)
            targets_Sk = targets_Sk / torch.nansum(targets_Sk, dim=(-2, -1), keepdim=True)

        # Compute spectral Sk divergence
        div = torch.nansum(targets_Sk * torch.log(torch.clamp(targets_Sk / predictions_Sk, min=1e-9)))
        return div
    

class SpectralRes(nn.Module):
    """
    Compute Spectral residual 
    (1) Validation mode: targeting specific top-k percentile wavenumber (higher k means higher frequency) is permissible
    (2) Training mode: computing metric along the entire wavenumber since some operation e.g., binning is nonautograd-able
    """
    def __init__(
        self,   
        percentile=0.9,
        input_shape=(121,240),
        is_train=True
    ):
        
        super(SpectralRes, self).__init__()
        
        self.percentile = percentile
        self.is_train = is_train
        
        # Compute the discrete Fourier Transform sample frequencies for a signal of size
        nx, ny = input_shape
        kx = torch.fft.fftfreq(nx) * nx
        ky = torch.fft.fftfreq(ny) * ny
        kx, ky = torch.meshgrid(kx, ky)
        
        self.k = torch.sqrt(kx**2 + ky**2).reshape(-1).to(config.device)
        self.k_low = 0.5
        self.k_upp = torch.max(self.k)
        self.k_nbin = torch.arange(self.k_low, torch.max(self.k), 1).size(0)
        
        # Get percentile index
        self.k_percentile_idx = int(self.k_nbin * self.percentile)
        
    def forward(self, predictions, targets):
        
        predictions = predictions.reshape(predictions.shape[0], -1, predictions.shape[-2], predictions.shape[-1])
        targets = targets.reshape(targets.shape[0], -1, targets.shape[-2], targets.shape[-1])
        
        assert predictions.shape[1] == targets.shape[1]
        nc = predictions.shape[1]
        
        # Handling missing values in predictions
        pred_means = torch.nanmean(predictions, dim=(-2, -1), keepdim=True)
        predictions = torch.where(torch.isnan(predictions), pred_means, predictions)
        
        # Compute along mini-batch
        predictions, targets = torch.nanmean(predictions, dim=0), torch.nanmean(targets, dim=0)
        
        # Transform prediction and targets onto the Fourier space and compute the power
        predictions_power, targets_power = torch.fft.fft2(predictions), torch.fft.fft2(targets)
        predictions_power, targets_power = torch.abs(predictions_power)**2, torch.abs(targets_power)**2

        ## If validation, we can target specific quantiles by binning and sorting
        if not self.is_train:
            predictions_Sk = torchist.histogram(self.k.repeat(nc), self.k_nbin, self.k_low, self.k_upp, weights=predictions_power) \
                            / torchist.histogram(self.k.repeat(nc), self.k_nbin, self.k_low, self.k_upp)

            targets_Sk = torchist.histogram(self.k.repeat(nc), self.k_nbin, self.k_low, self.k_upp, weights=targets_power) \
                        / torchist.histogram(self.k.repeat(nc), self.k_nbin, self.k_low, self.k_upp)
            
            # Extract top-k percentile wavenumber and its corresponding power spectrum
            predictions_Sk = predictions_Sk[self.k_percentile_idx:]
            targets_Sk = targets_Sk[self.k_percentile_idx:]
            
            # Normalize as pdf along ordered k
            predictions_Sk = predictions_Sk / torch.nansum(predictions_Sk)
            targets_Sk = targets_Sk / torch.nansum(targets_Sk)
        
        ## If training, compute the entire power spectrum 
        ## NOTE: targeting specific quantiles is yet to be implemented in autograd (i.e., binning operation)
        else:
            predictions_Sk, targets_Sk = predictions_power, targets_power
        
            # Normalize as pdf of each channel dimension
            predictions_Sk = predictions_Sk / torch.nansum(predictions_Sk, dim=(-2, -1), keepdim=True)
            targets_Sk = targets_Sk / torch.nansum(targets_Sk, dim=(-2, -1), keepdim=True)

        # Compute spectral Sk residual
        res = torch.sqrt(torch.nanmean(torch.square(predictions_Sk - targets_Sk)))
        return res
    

class CRPS(nn.Module):
    """Compute Continuous Ranked Probability Score (CRPS)
    """
    
    def __init__(self,
                 lat_adjusted=True):
        
        super(CRPS, self).__init__()
        
        self.lat_adjusted = lat_adjusted
        self.weights = get_adjusting_weights() if lat_adjusted else None
        
    def forward(self, predictions, targets):
        crps = []
        opts = dict(device=predictions.device, dtype=predictions.dtype)
        
        if self.lat_adjusted:
            predictions = self.weights.size(1) * (self.weights / torch.sum(self.weights)) * predictions 
            targets = self.weights.size(1) * (self.weights / torch.sum(self.weights)) * targets
            
        B, N, H, W = predictions.shape
        coords_pred = {"member": range(N), "lat": range(H), "lon": range(W)}
        coords_targ = {"lat": range(H), "lon": range(W)}
        
        # predictions = predictions.reshape((N, B, H, W))
        
        # predictions = predictions.sort(dim=0).values
        # diff = predictions[1:] - predictions[:-1]
        # weight = torch.arange(1, N, **opts) * torch.arange(N - 1, 0, -1, **opts)
        # weight = weight.reshape(weight.shape + (1,) * (diff.dim() - 1))
        
        # crps = torch.nanmean(torch.abs(predictions - targets), dim=0) - torch.nansum(diff * weight, dim=0) / N**2
        # crps = torch.nanmean(crps)
        
        for b in range(B):
            
            pred_xr = xr.DataArray(predictions[b].detach().cpu().numpy(), dims=["member", "lat", "lon"],  coords=coords_pred)
            targ_xr = xr.DataArray(targets[b].detach().cpu().numpy(), dims=["lat", "lon"], coords=coords_targ)
            
            # Compute CRPS for the current batch
            crps.append(crps_ensemble(targ_xr, pred_xr).mean().item())
            
        crps = torch.nanmean(torch.tensor(crps, **opts))

        return crps
    
class CRPSS(nn.Module):
    """Compute Continuous Ranked Probability Score (CRPS) Score
    """
    
    def __init__(self,
                 lat_adjusted=True):
        
        super(CRPSS, self).__init__()
        
        self.lat_adjusted = lat_adjusted
        self.weights = get_adjusting_weights() if lat_adjusted else None
        self.crps = CRPS(lat_adjusted=False) # latitude adjustment done only once...
        self.mae = MAE(lat_adjusted=False) # latitude adjustment done only once...
        self.normalization_file = {
                'era5': os.path.join(config.DATA_DIR, 'climatology', 'climatology_era5_spatial.zarr'),
                'lra5': os.path.join(config.DATA_DIR, 'climatology', 'climatology_lra5_spatial.zarr'),
                'oras5': os.path.join(config.DATA_DIR, 'climatology', 'climatology_oras5_spatial.zarr')
        }
        self.normalization = {
                'era5': xr.open_dataset(self.normalization_file['era5'], engine='zarr'),
                'lra5': xr.open_dataset(self.normalization_file['lra5'], engine='zarr'),
                'oras5': xr.open_dataset(self.normalization_file['oras5'], engine='zarr'),
        }

    def forward(self, predictions, targets, doys, param, source):
        
        opts = dict(device=predictions.device, dtype=predictions.dtype)
        
        # Get climatology
        clima_mean = torch.tensor(self.normalization[source]['mean'].sel(doy=doys, param=param).values).to(config.device)
        clima_sigma = torch.tensor(self.normalization[source]['sigma'].sel(doy=doys, param=param).values).to(config.device)
 
        if self.lat_adjusted:
            predictions = self.weights.size(1) * (self.weights / torch.sum(self.weights)) * predictions 
            targets = self.weights.size(1) * (self.weights / torch.sum(self.weights)) * targets
            clima_mean = self.weights.size(1) * (self.weights / torch.sum(self.weights)) * clima_mean
            clima_sigma = self.weights.size(1) * (self.weights / torch.sum(self.weights)) * clima_sigma
        
        B, N, H, W = predictions.shape
        coords_pred = {"member": range(N), "lat": range(H), "lon": range(W)}
        coords_targ = {"lat": range(H), "lon": range(W)}

        # Compute reference CRPS
        crps_ref = []
        for b in range(B):
            targ_xr = xr.DataArray(targets[b].detach().cpu().numpy(), dims=["lat", "lon"], coords=coords_targ)
            clima_m_xr = xr.DataArray(clima_mean[b].detach().cpu().numpy(), dims=["lat", "lon"], coords=coords_targ)
            clima_s_xr = xr.DataArray(clima_sigma[b].detach().cpu().numpy(), dims=["lat", "lon"], coords=coords_targ)
            crps_ref.append(crps_gaussian(targ_xr, clima_m_xr, clima_s_xr).mean().item())

        crps_ref = torch.nanmean(torch.tensor(crps_ref, **opts))
        
        # Compute forecast CRPS
        crps_for = self.crps(predictions, targets)
        
        # Compute CRPS Score
        crpss = 1 - crps_for / crps_ref

        return crpss


class Spread(nn.Module):
    """Compute Spread along the ensemble dimension
    """
    
    def __init__(self,
                 lat_adjusted=True):
        
        super(Spread, self).__init__()
        
        self.lat_adjusted = lat_adjusted
        self.weights = get_adjusting_weights() if lat_adjusted else None
        
    def forward(self, predictions, targets):
        
        if self.lat_adjusted:
            predictions = self.weights.size(1) * (self.weights / torch.sum(self.weights)) * predictions 
            
        spread = torch.nanmean(torch.std(predictions, dim=1))
    
        return spread
    

class SSR(nn.Module):
    """Compute spread/skill ratio
    """
    
    def __init__(self,
                 lat_adjusted=True):
        
        super(SSR, self).__init__()
        
        self.lat_adjusted = lat_adjusted
        self.weights = get_adjusting_weights() if lat_adjusted else None
        
    def forward(self, predictions, targets):
        
        skill = RMSE(lat_adjusted=self.lat_adjusted)
        spread = Spread(lat_adjusted=self.lat_adjusted)
        
        ssr = spread(predictions, targets) / skill(predictions.mean(axis=1), targets)
    
        return ssr
