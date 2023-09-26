DATA_DIR = '/burg/glab/projects/ChaosBench'
VARIABLE_LIST = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity']
PARAMS = ['z', 'q', 't', 'u', 'v', 'w']
PRESSURE_LEVELS = ['10', '50', '100','200', '300', '500','700', '850', '925','1000']
ERA5_YEARS = ['1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
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
UKMO_EXCEPTIONS = {
    '2016-03': "2016-03-01/2016-03-02/2016-03-03/2016-03-04/2016-03-05/2016-03-06/2016-03-07/2016-03-08/2016-03-09/2016-03-10/2016-03-11/2016-03-12/2016-03-13/2016-03-14/2016-03-15/2016-03-16/2016-03-17/2016-03-18/2016-03-19/2016-03-20/2016-03-21/2016-03-22/2016-03-23/2016-03-24/2016-03-25/2016-03-26/2016-03-27/2016-03-28/2016-03-30/2016-03-31",
    '2021-06': "2021-06-01/2021-06-02/2021-06-03/2021-06-04/2021-06-05/2021-06-06/2021-06-07/2021-06-09/2021-06-10/2021-06-11/2021-06-12/2021-06-13/2021-06-14/2021-06-15/2021-06-16/2021-06-17/2021-06-18/2021-06-19/2021-06-20/2021-06-21/2021-06-22/2021-06-23/2021-06-24/2021-06-25/2021-06-26/2021-06-27/2021-06-28/2021-06-29/2021-06-30",
    '2021-11': "2021-11-01/2021-11-02/2021-11-03/2021-11-04/2021-11-05/2021-11-06/2021-11-07/2021-11-08/2021-11-09/2021-11-10/2021-11-11/2021-11-12/2021-11-13/2021-11-14/2021-11-15/2021-11-16/2021-11-17/2021-11-18/2021-11-19/2021-11-20/2021-11-23/2021-11-25/2021-11-26/2021-11-27/2021-11-28/2021-11-29",
    '2022-08': "2022-08-01/2022-08-02/2022-08-03/2022-08-04/2022-08-05/2022-08-06/2022-08-07/2022-08-08/2022-08-09/2022-08-10/2022-08-11/2022-08-12/2022-08-14/2022-08-15/2022-08-16/2022-08-17/2022-08-19/2022-08-20/2022-08-21/2022-08-22/2022-08-24/2022-08-25/2022-08-26/2022-08-27/2022-08-28/2022-08-29/2022-08-30/2022-08-31",
    '2022-10': "2022-10-01/2022-10-02/2022-10-03/2022-10-04/2022-10-05/2022-10-06/2022-10-07/2022-10-08/2022-10-09/2022-10-10/2022-10-11/2022-10-12/2022-10-13/2022-10-14/2022-10-15/2022-10-16/2022-10-17/2022-10-18/2022-10-19/2022-10-20/2022-10-21/2022-10-22/2022-10-24/2022-10-25/2022-10-26/2022-10-27/2022-10-28/2022-10-29/2022-10-30/2022-10-31"
}

CMA_EXCEPTIONS = {
    '2016-02': "2016-02-01/to/2016-02-28",
    '2019-11': "2019-11-01/2019-11-02"
}
G_CONSTANT = 9.81
