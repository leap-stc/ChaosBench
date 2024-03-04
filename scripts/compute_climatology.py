import xarray as xr
from tqdm import tqdm
from pathlib import Path
import argparse
import numpy as np
import config
import re

import logging
logging.basicConfig(level=logging.INFO)
import warnings
warnings.filterwarnings("ignore")
    

def main(args):
    """
    Main driver to compute climatology of individual param/level pair
    Usage example: `python compute_climatology.py --dataset_name era5 --is_spatial False`
    """
    
    data_dir = Path(config.DATA_DIR) / args.dataset_name
    
    # Set output directory
    output_dir = Path(config.DATA_DIR) / 'climatology'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # If is_spatial flag is True, use a more complex way of computing climatology: sliding window
    if args.is_spatial:
        print('Computing climatology along spatial domain...')
        
        output_file = output_dir / f'climatology_{args.dataset_name}_spatial.zarr'
        
        ## Collect all files within the 30-year window
        years = np.arange(1994, 2024)
        doy = np.arange(0, 366)
        all_vars = dict()
        
        for year in tqdm(years):
            dataset_files = list()
            pattern = rf'.*{year}\d{{4}}\.zarr$'
            curr_files = list(data_dir.glob(f'*{year}*.zarr'))
            dataset_files.extend(
                [f for f in curr_files if re.match(pattern, str(f.name))]
            )

            dataset_files.sort()
        
            ## Compute climatology based on DOY
            for day_idx in doy:
                
                if day_idx < len(dataset_files):
                    ds = xr.open_dataset(dataset_files[day_idx], engine='zarr')
                    curr_var = ds[config.PARAMS].to_array().values
                
                else:
                    curr_var = np.full((len(config.PARAMS), len(config.PRESSURE_LEVELS), 121, 240), np.nan)
                                                   
                try:
                    all_vars[day_idx+1].append(curr_var)
                
                except:
                    all_vars[day_idx+1] = [curr_var]
                    
        
        all_vars = np.stack(list(all_vars.values()), axis=0) # Shape: (doy, year, param, level, lat, lon)
        
        # Aggregation
        climatology_mean = np.nanmean(all_vars, axis=(1))
        climatology_sigma = np.nanstd(all_vars, axis=(1))

        ds = xr.Dataset(
            {
                'mean': (('doy', 'param', 'level', 'lat', 'lon'), climatology_mean),
                'sigma': (('doy', 'param', 'level', 'lat', 'lon'), climatology_sigma),
            },

            coords = {
                'doy': doy + 1,
                'param': [str(param) for param in config.PARAMS],
                'level': [int(pressure_level) for pressure_level in config.PRESSURE_LEVELS],
                'lat': ds.latitude.values,
                'lon': ds.longitude.values
            }
        )
        
        
    else:
        print('Computing climatology including spatial domain...')
        
        output_file = output_dir / f'climatology_{args.dataset_name}.zarr'
        
        ## Collect all files
        dataset_files = list(data_dir.glob('*.zarr'))
        dataset_files.sort()

        # Collect values
        all_vars = list()

        for dataset_file in tqdm(dataset_files):

            ds = xr.open_dataset(dataset_file, engine='zarr')
            ds = ds[config.PARAMS]

            if args.dataset_name != 'era5':
                ds = ds.isel(step=0) # Only the first timestep is needed for instantenous forecast

            all_vars.append(
                ds.to_array().values
            )
            
        
        all_vars = np.array(all_vars) # Shape: (time, param, level, lat, lon)
        
        # Aggregation
        climatology_mean = np.nanmean(all_vars, axis=(0,3,4))
        climatology_sigma = np.nanstd(all_vars, axis=(0,3,4))

        ds = xr.Dataset(
            {
                'mean': (('param', 'level'), climatology_mean),
                'sigma': (('param', 'level'), climatology_sigma),
            },

            coords = {
                'param': [str(param) for param in config.PARAMS],
                'level': [int(pressure_level) for pressure_level in config.PRESSURE_LEVELS]
            }
        )

    # Save climatology
    ds.to_zarr(output_file)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', help='Provide the name of the dataset...')
    parser.add_argument('--is_spatial', help='If we perform climatology computation over the spatial grid...')
    
    args = parser.parse_args()
    main(args)
