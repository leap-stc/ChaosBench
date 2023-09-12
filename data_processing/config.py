DATA_DIR = '/burg/glab/projects/ChaosBench'
VARIABLE_LIST = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity']
PARAMS = ['z', 'q', 't', 'u', 'v', 'w']
PRESSURE_LEVELS = ['10', '50', '100','200', '300', '500','700', '850', '925','1000']
YEARS = ['2016', '2017', '2018', '2019', '2020', '2021', '2022']
MONTHS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
DAYS = ['01', '02', '03','04', '05', '06','07', '08', '09','10', '11', '12','13', '14', '15',
        '16', '17', '18','19', '20', '21','22', '23', '24','25', '26', '27','28', '29', '30','31']
STEPS = '0/24/48/72/96/120/144/168/192/216/240/264/288/312/336/360/384/408/432/456/480/504/528/552/576/600/624/648/672/696/720/744/768/792/816/840/864/888/912/936/960/984/1008/1032/1056'

S2S_CENTERS = {'cma': 'babj', 'ecmwf': 'ecmwf', 'ukmo': 'egrr', 'ncep': 'kwbc'} #
S2S_PARAM_LEVEL = {
    '130/131/132/156': '10/50/100/200/300/500/700/850/925/1000',
    '133': '200/300/500/700/850/925/1000',
    '135': '500'
}
G_CONSTANT = 9.81