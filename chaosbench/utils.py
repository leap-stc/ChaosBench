import xarray as xr
from pathlib import Path
import numpy as np
from datetime import datetime
from chaosbench import config

def convert_time(
    timestamp, 
    time_format='%Y-%m-%d'
):
    "Convert native datetimens object to specific format"
    
    timestamp_s = timestamp / 1e9  # Convert nanoseconds to seconds
    dt = datetime.utcfromtimestamp(timestamp_s)
    
    day = dt.strftime(time_format)
    
    return day

def denormalize(
    x, 
    param, 
    level, 
    dataset_name, 
    is_diff=False
):
    
    "Denormalize x given param/level and dataset name"
    
    # For some use-cases (eg. climatology, persistence forecasts), we use ERA5 as the benchmark climatology
    try:
        normalization_file = Path(config.DATA_DIR) / 's2s' / 'climatology' / f'climatology_{dataset_name}.zarr'
        normalization = xr.open_dataset(normalization_file, engine='zarr')
    except:
        normalization_file = Path(config.DATA_DIR) / 's2s' / 'climatology' / f'climatology_era5.zarr'
        normalization = xr.open_dataset(normalization_file, engine='zarr')
        
    normalization_mean = normalization['mean'].values
    normalization_sigma = normalization['sigma'].values
    
    param_idx, level_idx = config.PARAMS.index(param), config.PRESSURE_LEVELS.index(level)
    mean, sigma = normalization_mean[param_idx, level_idx], normalization_sigma[param_idx, level_idx]
    
    # Check if its a difference denormalization (eg. no +mean since it will cancel out)
    if is_diff:
        x = x * np.nanmean(sigma)
        
    else:
        x = (x * np.nanmean(sigma)) + np.nanmean(mean)
    
    return x
    
