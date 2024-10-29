import torch
from torch.utils.data import Dataset
from typing import List
from pathlib import Path
import glob
import xarray as xr
import numpy as np
from datetime import datetime
import re

from chaosbench import config, utils

class S2SObsDataset(Dataset):
    """
    Dataset object to handle input reanalysis.
    
    Params:
        years <List[int]>      : list of years to load and process,
        n_step <int>           : number of contiguous timesteps included in the data (default: 1)
        lead_time <int>        : delta_t ahead in time, useful for direct prediction (default: 1)
        atmos_vars <List[str]> : list of atmos variables to include (default: ALL)`
        land_vars <List[str]>  : list of land variables to include (default: empty)`
        ocean_vars <List[str]> : list of sea/ice variables to include (default: empty)`
        is_normalized <bool>   : flag to indicate whether we should perform normalization or not (default: True)
    """
    
    def __init__(
        self, 
        years: List[int],
        n_step: int = 1,
        lead_time: int = 1,
        atmos_vars: List[str] = [],
        land_vars: List[str] = [],
        ocean_vars: List[str] = [],
        is_normalized: bool = True
    ) -> None:
        
        self.data_dir = [
            Path(config.DATA_DIR) / 'era5',
            Path(config.DATA_DIR) / 'lra5',
            Path(config.DATA_DIR) / 'oras5'
        ]
        
        self.normalization_file = [
            Path(config.DATA_DIR) / 'climatology' / 'climatology_era5.zarr',
            Path(config.DATA_DIR) / 'climatology' / 'climatology_lra5.zarr',
            Path(config.DATA_DIR) / 'climatology' / 'climatology_oras5.zarr'
        ]
        
        self.years = [str(year) for year in years]
        self.n_step = n_step
        self.lead_time = lead_time
        self.atmos_vars = utils.get_param_level_list() if len(atmos_vars) == 0 else atmos_vars
        self.land_vars = land_vars
        self.ocean_vars = ocean_vars
        self.is_normalized = is_normalized
        
        # Subset files that match with patterns (eg. years specified)
        era5_files, lra5_files, oras5_files = list(), list(), list()
        for year in self.years:
            pattern = rf'.*{year}\d{{4}}\.zarr$'
            
            curr_files = [
                list(self.data_dir[0].glob(f'*{year}*.zarr')),
                list(self.data_dir[1].glob(f'*{year}*.zarr')),
                list(self.data_dir[2].glob(f'*{year}*.zarr'))
            ]
            
            era5_files.extend([f for f in curr_files[0] if re.match(pattern, str(f.name))])
            lra5_files.extend([f for f in curr_files[1] if re.match(pattern, str(f.name))])
            oras5_files.extend([f for f in curr_files[2] if re.match(pattern, str(f.name))])
        
        era5_files.sort(); lra5_files.sort(); oras5_files.sort()
        self.file_paths = [era5_files, lra5_files, oras5_files]
        
        # Subsetting
        self.era5_idx = [utils.get_param_level_idx(*param_level.split('-')) for param_level in self.atmos_vars]
        self.lra5_idx = [idx for idx, param in enumerate(config.LRA5_PARAMS) if param in self.land_vars]
        self.oras5_idx = [idx for idx, param in enumerate(config.ORAS5_PARAMS) if param in self.ocean_vars]
        
        # Retrieve climatology (i.e., mean and sigma) to normalize
        self.mean_era5 = xr.open_dataset(self.normalization_file[0], engine='zarr')['mean'].values[self.era5_idx, np.newaxis, np.newaxis]
        self.mean_lra5 = xr.open_dataset(self.normalization_file[1], engine='zarr')['mean'].values[self.lra5_idx, np.newaxis, np.newaxis]
        self.mean_oras5 = xr.open_dataset(self.normalization_file[2], engine='zarr')['mean'].values[self.oras5_idx, np.newaxis, np.newaxis]
        
        self.sigma_era5 = xr.open_dataset(self.normalization_file[0], engine='zarr')['sigma'].values[self.era5_idx, np.newaxis, np.newaxis]
        self.sigma_lra5 = xr.open_dataset(self.normalization_file[1], engine='zarr')['sigma'].values[self.lra5_idx, np.newaxis, np.newaxis]
        self.sigma_oras5 = xr.open_dataset(self.normalization_file[2], engine='zarr')['sigma'].values[self.oras5_idx, np.newaxis, np.newaxis]
        

    def __len__(self):
        data_length = len(self.file_paths[0]) - self.n_step - self.lead_time
        return data_length

    def __getitem__(self, idx):
        step_indices = [idx] + [target_idx for target_idx in range(idx + self.lead_time, idx + self.lead_time + self.n_step)]
        
        era5_data, lra5_data, oras5_data = list(), list(), list()
        
        for step_idx in step_indices:
            
            # Process era5
            era5_data.append(xr.open_dataset(self.file_paths[0][step_idx], engine='zarr')[config.PARAMS].to_array().values)
            
            # Process lra5
            if len(self.land_vars) > 0:
                lra5_data.append(xr.open_dataset(self.file_paths[1][step_idx], engine='zarr')[self.land_vars].to_array().values)
            
            # Process oras5
            if len(self.ocean_vars) > 0:
                oras5_data.append(xr.open_dataset(self.file_paths[2][step_idx], engine='zarr')[self.ocean_vars].to_array().values)
        
        # Permutation / reshaping
        era5_data, lra5_data, oras5_data = np.array(era5_data), np.array(lra5_data), np.array(oras5_data)
        era5_data = era5_data.reshape(era5_data.shape[0], -1, era5_data.shape[-2], era5_data.shape[-1]) # Merge (param, level) dims
        era5_data = era5_data[:, self.era5_idx] # Subset selected
        
        # Normalize
        if self.is_normalized:
            era5_data = (era5_data - self.mean_era5[np.newaxis, :, :, :]) / self.sigma_era5[np.newaxis, :, :, :]
            lra5_data = (lra5_data - self.mean_lra5[np.newaxis, :, :, :]) / self.sigma_lra5[np.newaxis, :, :, :]
            oras5_data = (oras5_data - self.mean_oras5[np.newaxis, :, :, :]) / self.sigma_oras5[np.newaxis, :, :, :]
        
        # Concatenate along parameter dimension, only if they are specified (i.e., non-empty)
        data = [t for t in [torch.tensor(era5_data), torch.tensor(lra5_data), torch.tensor(oras5_data)] if t.nelement() > 0]
        data = torch.cat(data, dim=1)
        
        x, y = data[0].float(), data[1:].float()
        timestamp = xr.open_dataset(self.file_paths[0][idx], engine='zarr').time.values.item()

        return timestamp, x, y
    
    
class S2SEvalDataset(Dataset):
    """
    Dataset object to load evaluation benchmarks.
    
    Params:
        s2s_name <str>        : center name where evaluation is going to be performed
        years <List[int]>     : list of years to load and process
        is_ensemble <bool>    : indicate whether to use control or perturbed ensemble forecasts
        is_normalized <bool>  : flag to indicate whether we should perform normalization or not (default: True)
    """
    
    def __init__(
        self, 
        s2s_name: str,
        years: List[int],
        is_ensemble = False,
        is_normalized = True
    ) -> None:
        
        assert s2s_name in list(config.S2S_CENTERS.keys())
        
        self.s2s_name = s2s_name
        self.data_dir = Path(config.DATA_DIR) / f'{self.s2s_name}_ensemble' if is_ensemble else Path(config.DATA_DIR) / self.s2s_name 
        self.normalization_file = Path(config.DATA_DIR) / 'climatology' / f'climatology_{self.s2s_name}.zarr'
        
        # Check if years specified are within valid bounds
        self.years = years
        self.years = [str(year) for year in self.years]
        assert set(self.years).issubset(set(config.YEARS))
        
        self.is_ensemble = is_ensemble
        self.is_normalized = is_normalized

        # Subset files that match with patterns (eg. years specified)
        file_paths = list()
        
        for year in self.years:
            pattern = rf'.*{year}\d{{4}}\.zarr$'
            curr_files = list(self.data_dir.glob(f'*{year}*.zarr'))
            file_paths.extend(
                [f for f in curr_files if re.match(pattern, str(f.name))]
            )
            
        # Subset files that match with patterns (eg. years specified)
        self.file_paths = file_paths
        self.file_paths.sort()
        
        # Retrieve climatology to normalize
        self.normalization = xr.open_dataset(self.normalization_file, engine='zarr')
        self.normalization_mean = self.normalization['mean'].values[:, np.newaxis, np.newaxis]
        self.normalization_sigma = self.normalization['sigma'].values[:, np.newaxis, np.newaxis]
        

    def __len__(self):
        return (len(self.file_paths) - config.N_STEPS)

    def __getitem__(self, idx):
        data = xr.open_dataset(self.file_paths[idx], engine='zarr')
        data = data[config.PARAMS].to_array().values
        
        if self.is_ensemble:
            data = torch.tensor(data).permute((2, 1, 0, 3, 4, 5)) # Shape: (step, ensem, param, level, lat, lon)
            data = data.reshape(data.shape[0], data.shape[1], -1, data.shape[-2], data.shape[-1]) # Shape: (step, ensem, param*level, lat, lon)
            
            if self.is_normalized:
                data = (data - self.normalization_mean[np.newaxis, np.newaxis, :, :, :]) / self.normalization_sigma[np.newaxis, np.newaxis, :, :, :] # Normalize

            x, y = data[0], data[1:]
            timestamp = xr.open_dataset(self.file_paths[idx], engine='zarr').time.values.item()
        
        else:
            data = torch.tensor(data).permute((1, 0, 2, 3, 4)) # Shape: (step, param, level, lat, lon)
            data = data.reshape(data.shape[0], -1, data.shape[-2], data.shape[-1]) # Shape: (step, param*level, lat, lon)
            
            if self.is_normalized:
                data = (data - self.normalization_mean[np.newaxis, :, :, :]) / self.normalization_sigma[np.newaxis, :, :, :] # Normalize

            x, y = data[0], data[1:]
            timestamp = xr.open_dataset(self.file_paths[idx], engine='zarr').time.values.item()
        
        return timestamp, x, y
