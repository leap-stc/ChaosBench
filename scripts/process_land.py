import argparse
import xarray as xr
from pathlib import Path
import config

import logging
logging.basicConfig(level=logging.INFO)

import cdsapi

def main(args):
    """
    Main driver to download LRA5 data based on individual variable
    See https://cds.climate.copernicus.eu/api-how-to on how to configure the API
    """
    RESOLUTION = str(args.resolution) 
    assert float(RESOLUTION) >= 0.25, 'Highest resolution is 0.25-degree, provide coarser one e.g., 1.5'

    # Initialize CDS API
    c = cdsapi.Client()
    
    # Set output directory
    output_dir = Path(config.DATA_DIR) / 'lra5'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download the corresponding data based on year/month
    for year in config.ERA5_YEARS:
        
        for month in config.MONTHS:
            
            logging.info(f'Downloading {year}/{month}...')
            
            output_file = output_dir / f'lra5_full_{RESOLUTION}deg_{year}{month}.nc'
            processed_sample_file = output_dir / f'lra5_full_{RESOLUTION}deg_{year}{month}01.zarr'
            
            # Skip downloading if file exists 
            if processed_sample_file.exists():
                continue
                
            else:
            
                c.retrieve(
                    'reanalysis-era5-land',
                    {
                        'variable': config.LRA5_LIST,
                        'year': year,
                        'month': [month],
                        'day': config.DAYS,
                        'time': '00:00',
                        'grid': [RESOLUTION, RESOLUTION],
                        'format': 'netcdf',
                    },
                    output_file)
                
                # Break down into daily .zarr (cloud-optimized)
                ds = xr.open_dataset(output_file)
                ds = ds.fillna(0)
                n_timesteps = len(ds.time)
                
                ## list() over multiple days
                for n_idx in range(n_timesteps):
                    subset_ds = ds.isel(time=n_idx)
                    yy, mm, dd = ds.time[n_idx].dt.strftime('%Y-%m-%d').item().split('-')
                    output_daily_file = output_dir / f'lra5_full_{RESOLUTION}deg_{yy}{mm}{dd}.zarr'
                    subset_ds.to_zarr(output_daily_file)
                
                output_file.unlink()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', default='1.5', help='Provide the resolution of preference, e.g., 1.5 for 1.5-degree...')
    
    args = parser.parse_args()
    main(args)
