from ecmwfapi import ECMWFDataServer
from pathlib import Path
import xarray as xr
import argparse
import calendar
import config

import logging
logging.basicConfig(level=logging.INFO)

def main(args):
    """
    Main driver to download ERA5 data based on individual variable
    Usage example: `python download_s2s.py --s2s_name ncep`
    """
    assert args.s2s_name in list(config.S2S_CENTERS.keys())
    center_id = config.S2S_CENTERS[args.s2s_name]
    
    # Initialize ECMWF API
    server = ECMWFDataServer()
    
    # Set output directory
    output_dir = Path(config.DATA_DIR) / 's2s' / args.s2s_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download the corresponding data based on year/month
    for year in config.YEARS:
        
        for month in config.MONTHS:
            
            logging.info(f'Downloading {year}/{month}...')
    
            # Skip downloading if file exists 
            output_file = output_dir / f'{args.s2s_name}_full_1.5deg_{year}{month}.nc'

            if output_file.exists():
                continue

            else:

                num_days = calendar.monthrange(int(year), int(month))[1]
                temp_files = list()

                for i, (param, level) in enumerate(config.S2S_PARAM_LEVEL.items()):

                    temp_file = output_dir / f"temp_{args.s2s_name}_{i}.grib"

                    server.retrieve({
                        "class": "s2",
                        "dataset": "s2s",
                        "date": f"{year}-{month}-01/to/{year}-{month}-{num_days}",
                        "expver": "prod",
                        "levelist": level,
                        "levtype": "pl",
                        "model": "glob",
                        "origin": center_id,
                        "param": param,
                        "step": config.STEPS,
                        "stream": "enfo",
                        "time": "00:00:00",
                        "type": "cf",
                        "target": str(temp_file)
                    })

                    temp_files.append(temp_file)

                # Combine and post-process each individual temp file (for now only 3)...
                assert len(temp_files) == 3
                temp_ds_0 = xr.open_dataset(temp_files[0], engine='cfgrib')
                temp_ds_1 = xr.open_dataset(temp_files[1], engine='cfgrib')
                temp_ds_2 = xr.open_dataset(temp_files[2], engine='cfgrib')
                temp_ds_2 = temp_ds_2.expand_dims({"isobaricInhPa": [temp_ds_2.isobaricInhPa.item()]})

                ## Merge and perform post-porocessing (eg. change variable/coordinate names to match ERA5)
                full_ds = xr.merge([temp_ds_0, temp_ds_1, temp_ds_2])
                full_ds = full_ds.transpose('time', 'step', 'isobaricInhPa', 'latitude', 'longitude')
                full_ds = full_ds.rename({'isobaricInhPa': 'level'})
                full_ds = full_ds.rename_vars({'gh': 'z'})

                ## Save as NetCDF file (plus compression)
                comp = dict(zlib=True, complevel=5)
                encoding = {var: comp for var in full_ds.data_vars}
                full_ds.to_netcdf(output_file, encoding=encoding)

                # Remove temp files (including idx files left behind by GRIB operation)
                for temp_file in temp_files:
                    temp_file.unlink()

                idx_files = list(output_dir.glob('*.idx'))
                for idx_file in idx_files:
                    idx_file.unlink()
                    
                # Break down into daily .zarr (cloud-optimized)
                ds = xr.open_dataset(output_file)
                n_timesteps = len(ds.time)
                
                for n_idx in range(n_timesteps):
                    subset_ds = ds.isel(time=n_idx)
                    yy, mm, dd = ds.time[n_idx].dt.strftime('%Y-%m-%d').item().split('-')
                    output_daily_file = output_dir / f'{args.s2s_name}_full_1.5deg_{yy}{mm}{dd}.zarr'
                    subset_ds.to_zarr(output_daily_file)
                
                output_file.unlink()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--s2s_name', help='Provide the name of the S2S center...')
    
    args = parser.parse_args()
    main(args)
