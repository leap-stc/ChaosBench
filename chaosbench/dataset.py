import torch
from torch.utils.data import Dataset
from typing import List
from pathlib import Path
import glob
import xarray as xr
import numpy as np
from datetime import datetime

from chaosbench import config

class S2SObsDataset(Dataset):
    """
    Dataset object to handle input ERA5 observations.
    
    Params:
        years <List[int]>: list of years to load and process
    
    """
    
    def __init__(
        self, 
        years: List[int]
    ) -> None:
        
        self.data_dir = Path(config.DATA_DIR) / 's2s' / 'era5'
        self.normalization_file = Path(config.DATA_DIR) / 's2s' / 'climatology' / 'climatology_era5.zarr'
        
        # Check if years specified are within valid bounds
        self.years = years
        self.years = [str(year) for year in self.years]
        
        assert set(self.years).issubset(set(config.YEARS))
        
        
        # Subset files that match with patterns (eg. years specified)
        file_paths = list()
        
        for year in self.years:
            
            file_paths.extend(
                list(
                    self.data_dir.glob(f'*{year}*.zarr')
                )
            )
            
        self.file_paths = file_paths
        self.file_paths.sort()
        
        # Retrieve climatology to normalize
        self.normalization = xr.open_dataset(self.normalization_file, engine='zarr')
        self.normalization_mean = self.normalization['mean'].values[:, :, np.newaxis, np.newaxis]
        self.normalization_sigma = self.normalization['sigma'].values[:, :, np.newaxis, np.newaxis]
        

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = xr.open_dataset(self.file_paths[idx], engine='zarr') # load data
        timestamp = data.time.values.item() # retrieve timestamp
        
        data = data[config.PARAMS].to_array().values # convert to array
        data = (data - self.normalization_mean) / self.normalization_sigma # normalize
        data = torch.tensor(data) # load as tensor
        
        return timestamp, data
    
class S2SEvalDataset(Dataset):
    """
    Dataset object to handle evaluation benchmarks.
    
    Params:
        s2s_name str: center name where evaluation is going to be performed
    
    """
    
    def __init__(
        self, 
        s2s_name: str
    ) -> None:
        
        assert s2s_name in list(config.S2S_CENTERS.keys())
        
        self.s2s_name = s2s_name
        self.data_dir = Path(config.DATA_DIR) / 's2s' / self.s2s_name
        self.normalization_file = Path(config.DATA_DIR) / 's2s' / 'climatology' / f'climatology_{self.s2s_name}.zarr'
        self.years = config.YEARS
        
        
        # Subset files that match with patterns (eg. years specified)
        self.file_paths = list(self.data_dir.glob(f'*.zarr'))
        self.file_paths.sort()
        
        # # Retrieve climatology to normalize
        self.normalization = xr.open_dataset(self.normalization_file, engine='zarr')
        self.normalization_mean = self.normalization['mean'].values[:, np.newaxis, :, np.newaxis, np.newaxis]
        self.normalization_sigma = self.normalization['sigma'].values[:, np.newaxis, :, np.newaxis, np.newaxis]
        

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = xr.open_dataset(self.file_paths[idx], engine='zarr') # load data (file store instead of file)
        timestamp = data.time.values.item() # retrieve timestamp
        
        data = data[config.PARAMS].to_array().values # convert to array
        data = (data - self.normalization_mean) / self.normalization_sigma # normalize
        data = torch.tensor(data) # load as tensor of shape (param, step, level, lat, lon)
        data = data.permute((1, 0, 2, 3, 4))
        
        return timestamp, data
