import xarray as xr
from pathlib import Path
import config

import logging
logging.basicConfig(level=logging.INFO)

import cdsapi

def main():
    """
    Main driver to download LRA5 data based on individual variable
    """
    # Initialize CDS API
    c = cdsapi.Client()
    
    # Set output directory
    output_dir = Path(config.DATA_DIR) / 'lra5'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download the corresponding data based on year/month
    for year in config.ERA5_YEARS:
        
        for month in config.MONTHS:
            
            logging.info(f'Downloading {year}/{month}...')
            
            output_file = output_dir / f'lra5_full_1.5deg_{year}{month}.nc'
            processed_sample_file = output_dir / f'lra5_full_1.5deg_{year}{month}01.zarr'
            
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
                        'grid': ['1.5', '1.5'],
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
                    output_daily_file = output_dir / f'lra5_full_1.5deg_{yy}{mm}{dd}.zarr'
                    subset_ds.to_zarr(output_daily_file)
                
                output_file.unlink()

if __name__ == "__main__":
    main()

