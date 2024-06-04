from pathlib import Path
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

################## CHANGE THIS TO YOUR OWN ##################
ABS_PATH = Path(__file__).resolve().parent.parent
DATA_DIR = ABS_PATH / 'data' 
#############################################################

PRESSURE_LEVELS = [10,   50,  100,  200,  300,  500,  700,  850,  925, 1000]
PARAMS = ['z', 'q', 't', 'u', 'v', 'w']

LRA5_PARAMS = ['asn', 'd2m', 'e', 'es', 'evabs', 'evaow', 'evatc', 'evavt', 'fal', 'lai_hv', 'lai_lv', 'pev', 'ro', 'rsn', 'sd', 'sde', 'sf', 'skt', 'slhf', 'smlt', 'snowc', 'sp', 'src', 'sro', 'sshf', 'ssr', 'ssrd', 'ssro', 'stl1', 'stl2', 'stl3', 'stl4', 'str', 'strd', 'swvl1', 'swvl2', 'swvl3', 'swvl4', 't2m', 'tp', 'tsn', 'u10', 'v10']

ORAS5_PARAMS = ['iicethic', 'iicevelu', 'iicevelv', 'ileadfra', 'so14chgt', 'so17chgt', 'so20chgt', 'so26chgt', 'so28chgt', 'sohefldo', 'sohtc300', 'sohtc700', 'sohtcbtm', 'sometauy', 'somxl010', 'somxl030', 'sosaline', 'sossheig', 'sosstsst', 'sowaflup', 'sozotaux']


S2S_CENTERS = {'cma': 'babj', 'ecmwf': 'ecmwf', 'ukmo': 'egrr', 'ncep': 'kwbc'} 
YEARS = ['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']

HEADLINE_VARS = ['t-850', 'z-500', 'q-700']
CLIMAX_VARS = ['z-50', 'z-500', 'z-700', 'z-850', 'z-925', 'u-50', 'u-500', 'u-700', 'u-850', 'u-925', 'v-50', 'v-500', 'v-700', 'v-850', 'v-925', 't-50', 't-500', 't-700', 't-850', 't-925', 'q-50', 'q-500', 'q-700', 'q-850', 'q-925']

N_STEPS = 45
