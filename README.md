# ChaosBench - A benchmark for long-term forecasting of chaotic systems
ChaosBench is a benchmark project to improve long-term forecasting of chaotic systems, in particular subseasonal-to-seasonal (S2S) weather. Current features include:

## 1. Benchmark and Dataset

- __Input:__ ERA5 Reanalysis (1979-2022)
    
- __Target:__ The following table indicates the 48 variables (channels) that are available for Physics-based models. Note that the __Input__ ERA5 observations contains __ALL__ fields, including the unchecked boxes:
        
    Parameters/Levels (hPa) | 1000 | 925 | 850 | 700 | 500 | 300 | 200 | 100 | 50 | 10
    :---------------------- | :----| :---| :---| :---| :---| :---| :---| :---| :--| :-|
    Geopotential height, z ($gpm$) | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; |  
    Specific humidity, q ($kg kg^{-1}$) | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &nbsp; | &nbsp; | &nbsp; |  
    Temperature, t ($K$) | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; |  
    U component of wind, u ($ms^{-1}$) | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; |  
    V component of wind, v ($ms^{-1}$) | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; |  
    Vertical velocity, w ($Pas^{-1}$) | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &check; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |  
    
- __Baselines:__
    - Physics-based models:
        - [x] NCEP: National Centers for Environmental Prediction
        - [x] CMA: China Meteorological Administration
        - [x] UKMO: UK Meteorological Office
        - [x] ECMWF: European Centre for Medium-Range Weather Forecasts
    - Data-driven models:
        - [x] UNet
        - [x] Resnet
        - [x] Lagged-Autoencoder
        - [x] Fourier Neural Operator (FNO)
        
## 2. Metrics
We divide our metrics into 2 classes: (1) ML-based, which cover evaluation used in conventional computer vision and forecasting tasks, (2) Physics-based, which are aimed to construct a more physically-faithful and explainable data-driven forecast.

- __Vision-based:__
    - [x] RMSE
    - [x] Bias
    - [x] Anomaly Correlation Coefficient (ACC)
    - [x] Structural Similarity Index (SSIM)
- __Physics-based:__
    - [x] Spectral Divergence
    - [x] Spectral Residual


## 3. Tasks
We presented two task, where the model still takes as inputs the __FULL__ 60 variables, but the benchmarking is done on either __ALL__ or __INDIVIDUAL__ target variable(s).

- __Task 1: Full Dynamics Prediction.__
It is aimed to target __ALL__ target channels simultaneously. This task is generally harder to perform but is useful to build a model that emulates the entire weather conditions.

- __Task 2: Sparse Dynamics Prediction.__
It is aimed to target an __INDIVIDUAL__ target channel. This task is useful to build long-term forecasting model for specific variables, such as surface temperature (t-1000) or surface humidity (q-1000). 

## 4. Getting Started
You can learn more about how to use our benchmark product through the following Jupyter notebooks under the `notebooks` directory. It covers topics ranging from:
- `01*_dataset_exploration`
- `02*_modeling`
- `03*_training`
- `04*_evaluation`
