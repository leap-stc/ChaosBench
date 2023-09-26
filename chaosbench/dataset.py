import torch
from torch.utils.data import Dataset
from typing import List
from pathlib import Path
import glob
import xarray as xr
import numpy as np
from datetime import datetime
import re

from chaosbench import config

class S2SObsDataset(Dataset):
    """
    Dataset object to handle input ERA5 observations.
    
    Params:
        years <List[int]>: list of years to load and process,
        time_step <int>  : number of time-lag (default = 1)
        is_val <bool>    : if the mode is validation (y is going to be n-step into the future)
    
    """
    
    def __init__(
        self, 
        years: List[int],
        time_step: int = 1,
        is_val: bool = False
    ) -> None:
        
        self.data_dir = Path(config.DATA_DIR) / 's2s' / 'era5'
        self.normalization_file = Path(config.DATA_DIR) / 's2s' / 'climatology' / 'climatology_era5.zarr'
        self.time_step = time_step
        self.is_val = is_val
        
        # Check if years specified are within valid bounds
        self.years = years
        self.years = [str(year) for year in self.years]
        
        # Subset files that match with patterns (eg. years specified)
        file_paths = list()
        
        for year in self.years:
            pattern = rf'.*{year}\d{{4}}\.zarr$'
            curr_files = list(self.data_dir.glob(f'*{year}*.zarr'))
            file_paths.extend(
                [f for f in curr_files if re.match(pattern, str(f.name))]
            )
            
        self.file_paths = file_paths
        self.file_paths.sort()
        
        # Retrieve climatology to normalize
        self.normalization = xr.open_dataset(self.normalization_file, engine='zarr')
        self.normalization_mean = self.normalization['mean'].values[:, :, np.newaxis, np.newaxis]
        self.normalization_sigma = self.normalization['sigma'].values[:, :, np.newaxis, np.newaxis]
        

    def __len__(self):
        data_length = (len(self.file_paths) - config.N_STEPS) if self.is_val else (len(self.file_paths) - self.time_step)
        return data_length

    def __getitem__(self, idx):
        
        # Process x
        x = xr.open_dataset(self.file_paths[idx], engine='zarr') # load data
        timestamp = x.time.values.item() # retrieve current input timestamp
        
        x = x[config.PARAMS].to_array().values # convert to array
        x = (x - self.normalization_mean) / self.normalization_sigma
        x = torch.tensor(x)
        
        # Process y
        if self.is_val:
            ## if under evaluation, it will return n_step future observations
            y = list() 
            for step_idx in range(config.N_STEPS):
                y.append(xr.open_dataset(self.file_paths[idx + step_idx], engine='zarr'))
            
            y = xr.concat(y, dim='step')
            y = y[config.PARAMS].to_array().values
            y = (y - self.normalization_mean[:, np.newaxis, :, :, :]) / self.normalization_sigma[:, np.newaxis, :, :, :]
            y = torch.tensor(y)
            y = y.permute((1, 0, 2, 3, 4)) # to shape (step, param, level, lat, lon)
            
            return timestamp, y
                
        else:
            ## Otherwhise, just the next time_step
            y = xr.open_dataset(self.file_paths[idx + self.time_step], engine='zarr')
            y = y[config.PARAMS].to_array().values
            y = (y - self.normalization_mean) / self.normalization_sigma
            y = torch.tensor(y)
        
            return timestamp, x, y
    
class S2SEvalDataset(Dataset):
    """
    Dataset object to handle evaluation benchmarks.
    
    Params:
        s2s_name str: center name where evaluation is going to be performed
    
    """
    
    def __init__(
        self, 
        s2s_name: str,
        years: List[int],
        is_val: bool = False
    ) -> None:
        
        assert s2s_name in list(config.S2S_CENTERS.keys())
        
        self.s2s_name = s2s_name
        self.data_dir = Path(config.DATA_DIR) / 's2s' / self.s2s_name
        self.normalization_file = Path(config.DATA_DIR) / 's2s' / 'climatology' / f'climatology_{self.s2s_name}.zarr'
        
        # Check if years specified are within valid bounds
        self.years = years
        self.years = [str(year) for year in self.years]
        assert set(self.years).issubset(set(config.YEARS))
        
        
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
        self.normalization_mean = self.normalization['mean'].values[:, :, np.newaxis, np.newaxis]
        self.normalization_sigma = self.normalization['sigma'].values[:, :, np.newaxis, np.newaxis]
        

    def __len__(self):
        return (len(self.file_paths) - config.N_STEPS)

    def __getitem__(self, idx):
        x = xr.open_dataset(self.file_paths[idx], engine='zarr') # load data 
        timestamp = x.time.values.item() # retrieve input timestamp
        
        x = x[config.PARAMS].to_array().values # convert to array
        x = (x - self.normalization_mean[:, np.newaxis, :, :, :]) / self.normalization_sigma[:, np.newaxis, :, :, :]
        x = torch.tensor(x) 
        x = x.permute((1, 0, 2, 3, 4)) # to shape (step, param, level, lat, lon)
        
        return timestamp, x
