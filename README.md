# ChaosBench - A benchmark for long-term forecasting of chaotic systems
ChaosBench is a framework to improve long-term forecasting of chaotic systems, including subseasonal-to-seasonal (S2S) climate systems and many other nonlinear toy problems, such as Rayleigh-Benard, Quasi-Geostrophic flow, Kelvin-Helmholtz. Our design paradigm revolves around __modularity__ and __extensibility__ so that you can easily separate modeling from benchmarking for instance, and to build on existing pieces with ease. Current features include:

## 1. Benchmark and Dataset

- __Subseasonal-to-seasonal (S2S) climate forecast__
    - [x] Observations: ERA5 Reanalysis (2016-2022)
    - Evaluation:
        - [x] NCEP
        - [ ] CMA
        - [ ] UKMO
        - [ ] ECMWF
    - Benchmark variables (ERA5 observations contains __ALL__ fields, including the unchecked boxes):
        
    Parameters/Levels (hPa) | 1000 | 925 | 850 | 700 | 500 | 300 | 200 | 100 | 50 | 10
    :---------------------- | :----| :---| :---| :---| :---| :---| :---| :---| :--| :-|
    Geopotential height, z ($gpm$) | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; |  
    Specific humidity, q ($kg kg^{-1}$) | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &nbsp; | &nbsp; | &nbsp; |  
    Temperature, t ($K$) | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; |  
    U component of wind, u ($ms^{-1}$) | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; |  
    V component of wind, v ($ms^{-1}$) | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; |  
    Vertical velocity, w ($Pas^{-1}$) | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &check; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |  
    
## 2. Models
- MLP

## 3. Metrics
- MSE
- RMSE

## 4. Tutorial
You can learn more about our benchmarking approach through our assortment of Jupyter notebooks under `notebooks`. It covers topics ranging from `dataset_exploration`, `modeling`, and `evaluation`. 

    

