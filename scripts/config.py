"""
This config file mainly deals with preprocessing of data, 
including accessing reanalysis and s2s center forecasts (control/perturbed)

"""

from pathlib import Path

################## CHANGE THIS TO YOUR OWN ##################
ABS_PATH = Path(__file__).resolve().parent.parent
DATA_DIR = ABS_PATH / 'data' 
#############################################################

ORAS5_LIST = ['depth_of_14_c_isotherm', 'depth_of_17_c_isotherm', 'depth_of_20_c_isotherm',
              'depth_of_26_c_isotherm', 'depth_of_28_c_isotherm', 'meridional_wind_stress',
              'mixed_layer_depth_0_01', 'mixed_layer_depth_0_03', 'net_downward_heat_flux',
              'net_upward_water_flux', 'ocean_heat_content_for_the_total_water_column', 'ocean_heat_content_for_the_upper_300m',
              'ocean_heat_content_for_the_upper_700m', 'sea_ice_concentration', 'sea_ice_meridional_velocity',
              'sea_ice_thickness', 'sea_ice_zonal_velocity', 'sea_surface_height',
              'sea_surface_salinity', 'sea_surface_temperature', 'zonal_wind_stress']

LRA5_LIST = ['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
            '2m_temperature', 'evaporation_from_bare_soil', 'evaporation_from_open_water_surfaces_excluding_oceans',
            'evaporation_from_the_top_of_canopy', 'evaporation_from_vegetation_transpiration', 'forecast_albedo',
            'leaf_area_index_high_vegetation', 'leaf_area_index_low_vegetation', 'potential_evaporation',
            'runoff', 'skin_reservoir_content', 'skin_temperature',
            'snow_albedo', 'snow_cover', 'snow_density',
            'snow_depth', 'snow_depth_water_equivalent', 'snow_evaporation',
            'snowfall', 'snowmelt', 'soil_temperature_level_1',
            'soil_temperature_level_2', 'soil_temperature_level_3', 'soil_temperature_level_4',
            'sub_surface_runoff', 'surface_latent_heat_flux', 'surface_net_solar_radiation',
            'surface_net_thermal_radiation', 'surface_pressure', 'surface_runoff',
            'surface_sensible_heat_flux', 'surface_solar_radiation_downwards', 'surface_thermal_radiation_downwards',
            'temperature_of_snow_layer', 'total_evaporation', 'total_precipitation',
            'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2', 'volumetric_soil_water_layer_3',
            'volumetric_soil_water_layer_4']

ERA5_LIST = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity']

ERA5_PARAMS = ['z', 'q', 't', 'u', 'v', 'w']
PRESSURE_LEVELS = ['10', '50', '100','200', '300', '500','700', '850', '925','1000']

LRA5_PARAMS = ['asn', 'd2m', 'e', 'es', 'evabs', 'evaow', 'evatc', 'evavt', 'fal', 'lai_hv', 'lai_lv', 'pev', 'ro', 'rsn', 'sd', 'sde', 'sf', 'skt', 'slhf', 'smlt', 'snowc', 'sp', 'src', 'sro', 'sshf', 'ssr', 'ssrd', 'ssro', 'stl1', 'stl2', 'stl3', 'stl4', 'str', 'strd', 'swvl1', 'swvl2', 'swvl3', 'swvl4', 't2m', 'tp', 'tsn', 'u10', 'v10']
ORAS5_PARAMS = ['iicethic', 'iicevelu', 'iicevelv', 'ileadfra', 'so14chgt', 'so17chgt', 'so20chgt', 'so26chgt', 'so28chgt', 'sohefldo', 'sohtc300', 'sohtc700', 'sohtcbtm', 'sometauy', 'somxl010', 'somxl030', 'sosaline', 'sossheig', 'sosstsst', 'sowaflup', 'sozotaux']


ERA5_YEARS = ['1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
YEARS = CF_YEARS = ['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
PF_YEARS = ['2022', '2023']

MONTHS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
DAYS = ['01', '02', '03','04', '05', '06','07', '08', '09','10', '11', '12','13', '14', '15',
        '16', '17', '18','19', '20', '21','22', '23', '24','25', '26', '27','28', '29', '30','31']
STEPS = '0/24/48/72/96/120/144/168/192/216/240/264/288/312/336/360/384/408/432/456/480/504/528/552/576/600/624/648/672/696/720/744/768/792/816/840/864/888/912/936/960/984/1008/1032/1056'

S2S_CENTERS = {'cma': 'babj', 'ecmwf': 'ecmwf', 'ukmo': 'egrr', 'ncep': 'kwbc'}
S2S_PARAM_LEVEL = {
    '130/131/132/156': '10/50/100/200/300/500/700/850/925/1000',
    '133': '200/300/500/700/850/925/1000',
    '135': '500'
}
UKMO_EXCEPTIONS = {
    '2016-03': "2016-03-01/2016-03-02/2016-03-03/2016-03-04/2016-03-05/2016-03-06/2016-03-07/2016-03-08/2016-03-09/2016-03-10/2016-03-11/2016-03-12/2016-03-13/2016-03-14/2016-03-15/2016-03-16/2016-03-17/2016-03-18/2016-03-19/2016-03-20/2016-03-21/2016-03-22/2016-03-23/2016-03-24/2016-03-25/2016-03-26/2016-03-27/2016-03-28/2016-03-30/2016-03-31",
    '2021-06': "2021-06-01/2021-06-02/2021-06-03/2021-06-04/2021-06-05/2021-06-06/2021-06-07/2021-06-09/2021-06-10/2021-06-11/2021-06-12/2021-06-13/2021-06-14/2021-06-15/2021-06-16/2021-06-17/2021-06-18/2021-06-19/2021-06-20/2021-06-21/2021-06-22/2021-06-23/2021-06-24/2021-06-25/2021-06-26/2021-06-27/2021-06-28/2021-06-29/2021-06-30",
    '2021-11': "2021-11-01/2021-11-02/2021-11-03/2021-11-04/2021-11-05/2021-11-06/2021-11-07/2021-11-08/2021-11-09/2021-11-10/2021-11-11/2021-11-12/2021-11-13/2021-11-14/2021-11-15/2021-11-16/2021-11-17/2021-11-18/2021-11-19/2021-11-20/2021-11-23/2021-11-25/2021-11-26/2021-11-27/2021-11-28/2021-11-29",
    '2022-08': "2022-08-01/2022-08-02/2022-08-03/2022-08-04/2022-08-05/2022-08-06/2022-08-07/2022-08-08/2022-08-09/2022-08-10/2022-08-11/2022-08-14/2022-08-15/2022-08-16/2022-08-17/2022-08-19/2022-08-20/2022-08-21/2022-08-22/2022-08-24/2022-08-25/2022-08-26/2022-08-27/2022-08-28/2022-08-29/2022-08-30/2022-08-31",
    '2022-10': "2022-10-01/2022-10-02/2022-10-03/2022-10-04/2022-10-05/2022-10-06/2022-10-07/2022-10-08/2022-10-09/2022-10-10/2022-10-11/2022-10-12/2022-10-13/2022-10-14/2022-10-15/2022-10-16/2022-10-17/2022-10-18/2022-10-19/2022-10-20/2022-10-21/2022-10-22/2022-10-24/2022-10-25/2022-10-26/2022-10-27/2022-10-28/2022-10-29/2022-10-30/2022-10-31",
    '2022-11': "2022-11-01/2022-11-02/2022-11-03/2022-11-04/2022-11-05/2022-11-06/2022-11-07/2022-11-08/2022-11-09/2022-11-10/2022-11-11/2022-11-12/2022-11-13/2022-11-14/2022-11-15/2022-11-16/2022-11-17/2022-11-19/2022-11-20/2022-11-21/2022-11-22/2022-11-23/2022-11-24/2022-11-25/2022-11-27/2022-11-28/2022-11-29/2022-11-30",
    '2023-12': "2023-12-01/2023-12-02/2023-12-04/2023-12-05/2023-12-06/2023-12-07/2023-12-08/2023-12-09/2023-12-10/2023-12-11/2023-12-12/2023-12-13/2023-12-14/2023-12-15/2023-12-16/2023-12-17/2023-12-18/2023-12-19/2023-12-20/2023-12-21/2023-12-22/2023-12-23/2023-12-24/2023-12-25/2023-12-26/2023-12-27/2023-12-28/2023-12-29/2023-12-30/2023-12-31"
}

CMA_EXCEPTIONS = {
    '2016-02': "2016-02-01/to/2016-02-28",
    '2019-11': "2019-11-01/2019-11-02"
}

ECMWF_EXCEPTIONS = {
    '2023-07': "2023-07-01/to/2023-07-31",
    '2023-08': "2023-08-01/to/2023-08-31",
    '2023-09': "2023-09-01/to/2023-09-30",
    '2023-10': "2023-10-01/to/2023-10-31",
    '2023-11': "2023-11-01/to/2023-11-30",
    '2023-12': "2023-12-01/to/2023-12-31"
}

ENSEMBLE_NUMBERS = {
    'ecmwf': "1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/31/32/33/34/35/36/37/38/39/40/41/42/43/44/45/46/47/48/49/50",
    'ukmo': "1/2/3",
    'ncep': "1/2/3/4/5/6/7/8/9/10/11/12/13/14/15",
    'cma': "1/2/3"
}

G_CONSTANT = 9.81
