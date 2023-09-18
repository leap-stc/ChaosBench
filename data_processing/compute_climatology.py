import xarray as xr
from tqdm import tqdm
from pathlib import Path
import argparse
import numpy as np
import config

import logging
logging.basicConfig(level=logging.INFO)

def main(args):
    """
    Main driver to compute climatology of individual param/level pair
    Usage example: `python compute_climatology.py --dataset_name era5`
    """
    
    # Set output directory
    output_dir = Path(config.DATA_DIR) / 's2s' / 'climatology'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir/ f'climatology_{args.dataset_name}.zarr'

    # Skip processing if file exists 
    if output_file.exists():
        logging.info(f'Climatology for {args.dataset_name} has been processed...')
        
    else:
    
        # Retrieve all files
        dataset_files = list((Path(config.DATA_DIR) / 's2s' / args.dataset_name).glob('*.zarr'))
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

        # Compute climatology
        all_vars = np.array(all_vars) # Shape: (time, param, level, lat, lon)
        climatology_mean = np.nanmean(all_vars, axis=(0,3,4))
        climatology_sigma = np.nanstd(all_vars, axis=(0,3,4))

        # Build Dataset object
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
    
    args = parser.parse_args()
    main(args)
