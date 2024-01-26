import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

######### CHANGE THIS TO YOUR OWN #########
DATA_DIR = '/burg/glab/projects/ChaosBench' 
###########################################

YEARS = ['2016', '2017', '2018', '2019', '2020', '2021', '2022']
PARAMS = ['z', 'q', 't', 'u', 'v', 'w']
PRESSURE_LEVELS = [10,   50,  100,  200,  300,  500,  700,  850,  925, 1000]
S2S_CENTERS = {'cma': 'babj', 'ecmwf': 'ecmwf', 'ukmo': 'egrr', 'ncep': 'kwbc'} 
HEADLINE_VARS = ['t-850', 'z-500', 'q-700']
N_STEPS = 45