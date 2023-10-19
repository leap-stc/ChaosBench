# ChaosBench - A benchmark for long-term forecasting of chaotic systems
ChaosBench is a benchmark project to improve long-term forecasting of chaotic systems, including subseasonal-to-seasonal (S2S) climate systems and other nonlinear dynamical problems including Rayleigh-Benard, Quasi-Geostrophic, Kelvin-Helmholtz, etc. Our design paradigm revolves around __modularity__ and __extensibility__ so that you can easily separate modeling from benchmarking for instance, and to build on existing pieces with ease. Current features include:

## 1. Benchmark and Dataset

- __(ChaosBench-S2S) Subseasonal-to-seasonal (S2S) climate forecast__
    - Observations: 
        - [x] ERA5 Reanalysis (1979-2022)
    - Physics-based Evaluation Benchmark:
        - [x] NCEP
        - [x] CMA
        - [x] UKMO
        - [x] ECMWF
    - Climatology: the long-term mean and sigma for ERA5, NCEP, CMA, UKMO, and ECMWF products are also available for (de)-normalization and the compute of metrics (eg. ACC)
    - Benchmark variables (ERA5 observations contains __ALL__ fields, including the unchecked boxes):
        
    Parameters/Levels (hPa) | 1000 | 925 | 850 | 700 | 500 | 300 | 200 | 100 | 50 | 10
    :---------------------- | :----| :---| :---| :---| :---| :---| :---| :---| :--| :-|
    Geopotential height, z ($gpm$) | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; |  
    Specific humidity, q ($kg kg^{-1}$) | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &nbsp; | &nbsp; | &nbsp; |  
    Temperature, t ($K$) | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; |  
    U component of wind, u ($ms^{-1}$) | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; |  
    V component of wind, v ($ms^{-1}$) | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; |  
    Vertical velocity, w ($Pas^{-1}$) | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &check; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |  
    
- TBD

## 2. Data-Driven Models
- CNN (UNet, ResNet)
- Generative (Autoencoder)
- Fourier Neural Operator

## 3. Metrics
- RMSE
- MAE
- Bias
- Anomaly Correlation Coefficient (ACC)
- Coefficient of Determination ($R^2$)
- Spectral Divergence (*NEW*)


## 4. Tutorial
You can learn more about our benchmarking approach through our assortment of Jupyter notebooks under `notebooks`. It covers topics ranging from 
- `01*_dataset_exploration`
- `02*_modeling`
- `03*_training`
- `04*_evaluation`

## 5. Experiments
We perform the following experiments:
- Comparing the forecasting performance of physical and data-driven models
- Evaluating the value of temporal information
- TBD
