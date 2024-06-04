import torch
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
    dataset_name, 
    is_diff=False
):
    
    "Denormalize x given param/level and dataset name"
    
    # For some use-cases (eg. climatology, persistence forecasts), we use ERA5 as the benchmark climatology
    try:
        normalization_file = Path(config.DATA_DIR) / 'climatology' / f'climatology_{dataset_name}.zarr'
        normalization = xr.open_dataset(normalization_file, engine='zarr')
    except:
        normalization_file = Path(config.DATA_DIR) / 'climatology' / f'climatology_era5.zarr'
        normalization = xr.open_dataset(normalization_file, engine='zarr')
    
    # Get their mean and sigma values
    mean = normalization['mean'].sel(param=param).values
    sigma = normalization['sigma'].sel(param=param).values
    
    # Check if its a difference denormalization (eg. no +mean since it will cancel out)
    if is_diff:
        x = x * np.nanmean(sigma)
        
    else:
        x = (x * np.nanmean(sigma)) + np.nanmean(mean)
    
    return x
    
def get_param_level_idx(param, level):
    """Given param and level, get flattended index especially for atmospheric dataset"""
    return config.PARAMS.index(param) * len(config.PRESSURE_LEVELS) + config.PRESSURE_LEVELS.index(level)

def flat_to_level(data):
    """Given flattened (param-level) to (param, level) dataset"""
    n_dims = len(data.shape)
    
    if n_dims == 3:
        P, H, W = data.shape
        return data.reshape(len(config.PARAMS), len(config.PRESSURE_LEVELS), H, W)
    
    elif n_dims == 4:
        B, P, H, W = data.shape
        return data.reshape(B, len(config.PARAMS), len(config.PRESSURE_LEVELS), H, W)
    
    elif n_dims == 5:
        B, S, P, H, W = data.shape
        return data.reshape(B, S, len(config.PARAMS), len(config.PRESSURE_LEVELS), H, W)
    
    else:
        B, S, N, P, H, W = data.shape
        return data.reshape(B, S, N, len(config.PARAMS), len(config.PRESSURE_LEVELS), H, W)
    
    
def get_doys_from_timestep(timestamps, lead_time=1):
    """
    Get climatology data given timestamps and lead_time
    
    Param:
        timestamps   : list of datetime[ns] object
        lead_time    : offset to apply to climatology doy (default: 1; i.e., target is 1-day ahead)
    """
    all_doys = list()
    
    for timestamp in timestamps:
        doy = datetime.utcfromtimestamp(timestamp.item() / 1e9).timetuple().tm_yday
        doys = torch.arange(doy, doy+config.N_STEPS-lead_time)
        doys = doys + lead_time
        all_doys.append(doys)
    
    return torch.stack(all_doys)

def get_param_level_list():
    return [f"{var}-{level}" for var in config.PARAMS for level in config.PRESSURE_LEVELS]
