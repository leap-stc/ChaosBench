import cdsapi

import xarray as xr
from pathlib import Path
import config
import tarfile
import numpy as np
import calendar
from scipy.interpolate import griddata

import logging
logging.basicConfig(level=logging.INFO)

import cdsapi

def main():
    """
    Main driver to download ORAS5 data based on individual variable
    """
    # Initialize CDS API
    c = cdsapi.Client()
    
    # Set output directory
    output_dir = Path(config.DATA_DIR) / 'oras5'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download the corresponding data based on year/month
    for year in config.ERA5_YEARS:
        
        for month in config.MONTHS:
            
            logging.info(f'Downloading {year}/{month}...')
            
            output_file = output_dir / f'oras5_full_1.5deg_{year}{month}.tar.gz'
            processed_sample_file = output_dir / f'oras5_full_1.5deg_{year}{month}01.zarr'
            
            
            # Skip downloading if file exists
            if processed_sample_file.exists():
                continue
                
            else:
                product_type = 'consolidated' if int(year) <= 2014 else 'operational'
    
                c.retrieve(
                    'reanalysis-oras5',
                    {
                        'product_type': product_type,
                        'vertical_resolution': 'single_level',
                        'variable': config.ORAS5_LIST,
                        'year': year,
                        'month': [month],
                        'format': 'tgz',
                    },
                    output_file)
            
                # Extract files and collect all netcdf file
                with tarfile.open(output_file, 'r:gz') as tar:
                    tar.extractall(path=output_dir)
                    
                nc_files = list(output_dir.glob('*.nc'))
                nc_files.sort()
                
                _, num_days = calendar.monthrange(int(year), int(month))
                all_ds = list()
                
                for nc_file in nc_files:
                    
                    # Process 0: Open each .nc file
                    ds = xr.open_dataset(nc_file)
                    ds['nav_lon'] = xr.where(ds['nav_lon'] < 0, ds['nav_lon'] + 360, ds['nav_lon'])
                    
                    # Process 1: Regridding to ERA5 target grid
                    new_lon, new_lat = np.meshgrid(np.linspace(0, 360, num=240, endpoint=False), np.linspace(-90, 90, num=121))
                    regridded_ds = xr.Dataset()
                    
                    for var in list(ds.data_vars)[:1]:
                        val, lat, lon = ds[var].values.flatten(), ds.nav_lat.values.flatten(), ds.nav_lon.values.flatten()
                        coords = np.array([lat, lon]).T
                        regridded_ds[var] = (
                            ('latitude', 'longitude'), 
                            griddata(coords, val, (new_lat, new_lon), method='linear')
                        )

                    regridded_ds = regridded_ds.assign_coords(
                        {"latitude": (("latitude",), new_lat[:, 0]),
                         "longitude": (("longitude",), new_lon[0, :])}
                    )
                    regridded_ds = regridded_ds.reindex(latitude=regridded_ds.latitude[::-1])

                    # Process 2: Handle NaN values
                    regridded_ds = regridded_ds.fillna(0)
                    
                    # Process 3: Merge
                    all_ds.append(regridded_ds)
                all_ds = xr.merge(all_ds)
                    
                # Process 4: Break down into daily .zarr (cloud-optimized)
                for n_idx in range(num_days):
                    output_daily_file = output_dir / f'oras5_full_1.5deg_{year}{month}{str(n_idx+1).zfill(2)}.zarr'
                    all_ds.to_zarr(output_daily_file)
                
                # Process 5: Remove intermediary files
                output_file.unlink()
                for nc_file in nc_files:
                    nc_file.unlink() 

if __name__ == "__main__":
    main()
