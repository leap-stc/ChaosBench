# Dataset Information

> __NOTE__: Hands-on exploration of the ChaosBench dataset in `notebooks/01a_s2s_data_exploration.ipynb`

1. __Input:__ ERA5 Reanalysis (1979-2023)
    
2. __Target:__ The following table indicates the 48 variables (channels) that are available for Physics-based models. Note that the __Input__ ERA5 observations contains __ALL__ fields, including the unchecked boxes:
        
    Parameters/Levels (hPa) | 1000 | 925 | 850 | 700 | 500 | 300 | 200 | 100 | 50 | 10
    :---------------------- | :----| :---| :---| :---| :---| :---| :---| :---| :--| :-|
    Geopotential height, z ($gpm$) | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; |  
    Specific humidity, q ($kg kg^{-1}$) | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &nbsp; | &nbsp; | &nbsp; |  
    Temperature, t ($K$) | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; |  
    U component of wind, u ($ms^{-1}$) | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; |  
    V component of wind, v ($ms^{-1}$) | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; |  
    Vertical velocity, w ($Pas^{-1}$) | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &check; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |  
    