import argparse
import subprocess
from pathlib import Path
import config

import numpy as np
import xarray as xr

def process_param_levels(ds):
    """Flatten variable/level and return list of available parameters"""

    flat_ds = {}

    for data_var in list(ds.data_vars):
        curr_ds = ds[data_var]
        data_levels = curr_ds.isobaricInhPa.values

        for data_level in data_levels:
            flat_varname = f'{data_var}-{int(data_level)}'
            flat_ds[flat_varname] = curr_ds.sel(isobaricInhPa=data_level).drop_vars('isobaricInhPa')

    return xr.Dataset(flat_ds)


def main(args):
    """
    Main driver to download S2S data-driven benchmark data based
    For now, we only process bi-weekly forecasts (decorrelated timescale)
    But the script provided here can be easily extended for any resolution, with minor modifications...
    The script relies on the excellent `ai-models` package maintained by the ECMWF

    README: to setup, follow instructions from https://github.com/ecmwf-lab/ai-models.
            you might also need to setup CDS API, follow instructions from https://cds.climate.copernicus.eu/how-to-api
    
    Usage example: 
        (1) Panguweather    :   `python process_sota.py --model_name panguweather --years 2022`
        (2) Graphcast       :   `python process_sota.py --model_name graphcast --years 2022`
        (3) FourcastNetV2   :   `python process_sota.py --model_name fourcastnetv2 --years 2022`
    """
    assert args.model_name in ['fourcastnetv2', 'panguweather', 'graphcast']
    
    output_dir = config.DATA_DIR / f'{args.model_name}'
    asset_dir = output_dir / 'assets'
    
    model_code = f'{args.model_name}-small' if args.model_name == 'fourcastnetv2' else args.model_name
    mm_dds = ['0101', '0115', '0201', '0215', '0301', '0315', '0401', '0415', '0501', '0515', '0601', '0615', 
              '0701', '0715', '0801', '0815', '0901', '0915', '1001', '1015', '1101', '1115', '1201', '1215']
    for year in args.years:
        year = int(year)
        
        # NOTE: Feel free to relax this monthly implementation to get e.g., daily forecasts
        # At present (2024): biweekly (decorrelated timescale)
        # The eval_sota.py script should be able to automatically handle different forecast frequency
        for mm_dd in mm_dds:
            output_daily_file = output_dir / f'{args.model_name}_full_1.5deg_{year}{mm_dd}.zarr'
            
            if not output_daily_file.exists():
                
                temp_file = output_dir / f'{args.model_name}_{year}{mm_dd}.grib'

                # Get prediction
                command = [
                    "ai-models", 
                    "--input", "cds", 
                    "--path", temp_file, 
                    "--assets", asset_dir, 
                    "--date", f"{year}{mm_dd}", 
                    "--time", "0000", 
                    "--lead-time", "1056", 
                    model_code
                ]

                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode != 0:
                    print(result.stderr)  # Print any error message

                # Process prediction
                dataset = xr.open_dataset(temp_file, backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}})
                dataset['z'] = dataset['z'] / config.G_CONSTANT # to gpm conversion
                dataset = process_param_levels(dataset)
                dataset = dataset.isel(step=slice(4, None, 4)) # every 24th-hour --> daily resolution
                dataset = dataset.coarsen(latitude=6, longitude=6, boundary='trim').mean()
                dataset = dataset.interp(latitude=np.linspace(dataset.latitude.values.max(), dataset.latitude.values.min(), 121))

                # Break down into daily .zarr (cloud-optimized)
                dataset.to_zarr(output_daily_file)

                # Post-process (cleaning up files)
                idx_files = list(temp_file.parent.glob(f"{temp_file.stem}.*.idx"))
                for idx_file in idx_files:
                    idx_file.unlink()

                temp_file.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help='Provide the name of the model, e.g., fourcastnetv2, graphcast, panguweather...')
    parser.add_argument('--years', nargs='+', help='Provide the years to evaluate on...')

    args = parser.parse_args()
    main(args)
