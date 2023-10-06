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
        n_step <int>     : number of contiguous timesteps included in the data (default: 1)
    
    """
    
    def __init__(
        self, 
        years: List[int],
        n_step: int = 1
    ) -> None:
        
        self.data_dir = Path(config.DATA_DIR) / 's2s' / 'era5'
        self.normalization_file = Path(config.DATA_DIR) / 's2s' / 'climatology' / 'climatology_era5.zarr'
        self.n_step = n_step
        
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
        data_length = len(self.file_paths) - self.n_step
        return data_length

    def __getitem__(self, idx):
        data = list() 
        for step_idx in range(self.n_step + 1):
            data.append(xr.open_dataset(self.file_paths[idx + step_idx], engine='zarr'))

        data = xr.concat(data, dim='step')
        data = data[config.PARAMS].to_array().values
        data = (data - self.normalization_mean[:, np.newaxis, :, :, :]) / self.normalization_sigma[:, np.newaxis, :, :, :]
        data = torch.tensor(data)
        data = data.permute((1, 0, 2, 3, 4)) # to shape (step, param, level, lat, lon)
        
        x = data[0]
        y = data[1:]
        
        timestamp = xr.open_dataset(self.file_paths[idx], engine='zarr').time.values.item()

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
        data = xr.open_dataset(self.file_paths[idx], engine='zarr') # load data 
        
        data = data[config.PARAMS].to_array().values # convert to array
        data = (data - self.normalization_mean[:, np.newaxis, :, :, :]) / self.normalization_sigma[:, np.newaxis, :, :, :]
        data = torch.tensor(data) 
        data = data.permute((1, 0, 2, 3, 4)) # to shape (step, param, level, lat, lon)
        
        x = data[0]
        y = data[1:]
        
        timestamp = xr.open_dataset(self.file_paths[idx], engine='zarr').time.values.item()
        
        return timestamp, x, y
