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
    Usage example: 
        (1) Scalar climatology for training  : `python compute_climatology.py --dataset_name era5 --is_spatial 0`
        (2) Proper DOY-spatial climatology   : `python compute_climatology.py --dataset_name era5 --is_spatial 1`
    """
    
    data_dir = Path(config.DATA_DIR) / args.dataset_name
    
    # Set output directory
    output_dir = Path(config.DATA_DIR) / 'climatology'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # If is_spatial flag is True, use a more complex way of computing climatology: day-of-year / sliding window
    if args.is_spatial:
        print('Computing climatology along spatial domain...')
        
        output_file = output_dir / f'climatology_{args.dataset_name}_spatial.zarr'
        
        ## Handling parameter list
        if args.dataset_name == 'era5':
            param_list = config.ERA5_PARAMS
            param_string = [f"{param}-{level}" for param in config.ERA5_PARAMS for level in config.PRESSURE_LEVELS]
        
        elif args.dataset_name == 'lra5':
            param_list = param_string = config.LRA5_PARAMS
        
        elif args.dataset_name == 'oras5':
            param_list = param_string = config.ORAS5_PARAMS
        
        else:
            raise NotImplementedError('climatology computation over spatial domain for this dataset is yet to be implemented...')

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
                    curr_var = ds[param_list].to_array().values
                    curr_var = curr_var.reshape(-1, curr_var.shape[-2], curr_var.shape[-1])

                else:
                    param_size = len(param_list) * len(config.PRESSURE_LEVELS) if args.dataset_name == 'era5' else len(param_list)
                    curr_var = np.full((param_size, 121, 240), np.nan)
                                                   
                try:
                    all_vars[day_idx+1].append(curr_var)
                
                except:
                    all_vars[day_idx+1] = [curr_var]
                    
        
        all_vars = np.stack(list(all_vars.values()), axis=0) # Shape: (doy, year, param*level, lat, lon)
        
        # Aggregation
        climatology_mean = np.nanmean(all_vars, axis=(1))
        climatology_sigma = np.nanstd(all_vars, axis=(1))
        
        # Climatology dataset construction
        ds = xr.Dataset(
            {
                'mean': (('doy', 'param', 'lat', 'lon'), climatology_mean),
                'sigma': (('doy', 'param', 'lat', 'lon'), climatology_sigma),
            },

            coords = {
                'doy': doy + 1,
                'param': param_string,
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
            
            # Handle datasets with pressure levels (eg. s2s centers + era5 atmos)
            if (args.dataset_name == 'era5') or (args.dataset_name in list(config.S2S_CENTERS.keys())):
                
                ds = ds[config.ERA5_PARAMS]
                
                # Only the first timestep is needed for instantenous forecast
                if args.dataset_name in list(config.S2S_CENTERS.keys()):
                    ds = ds.isel(step=0)
            
            # Handle datasets without pressure levels (eg. ocean + land reanalysis)
            else:
                
                PARAMS = config.ORAS5_PARAMS if args.dataset_name == 'oras5' else config.LRA5_PARAMS
                ds = ds[PARAMS]
                
            ds = ds.to_array().values
            all_vars.append(ds.reshape(-1, ds.shape[-2], ds.shape[-1]))
            
        
        all_vars = np.array(all_vars) # Shape (w/ levels): (time, param*level, lat, lon); (surface): (time, param, lat, lon)
        
        # Aggregation
        climatology_mean = np.nanmean(all_vars, axis=(0,2,3))
        climatology_sigma = np.nanstd(all_vars, axis=(0,2,3))
        
        if (args.dataset_name == 'oras5') or (args.dataset_name == 'lra5'):
            param_string = PARAMS
            
        else: 
            param_string = [f"{param}-{level}" for param in config.ERA5_PARAMS for level in config.PRESSURE_LEVELS] 
        
        # Climatology dataset construction
        ds = xr.Dataset(
            {
                'mean': (('param'), climatology_mean),
                'sigma': (('param'), climatology_sigma),
            },

            coords = {
                'param': param_string,
            }
        )

    # Save climatology
    ds.to_zarr(output_file)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', help='Provide the name of the dataset...')
    parser.add_argument('--is_spatial', type=int, help='If we perform climatology computation over the spatial grid...')
    
    args = parser.parse_args()
    main(args)
