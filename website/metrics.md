# Metrics

> __Note__: For all models logs, there will be a folder named `eval/`. This contains individual `.csv` files for each relevant metric (e.g., RMSE, SpecDiv).


We divide our metrics into 3 classes: (1) Deterministic-based, which cover evaluation used in conventional deterministic forecasting tasks, (2) Physics-based, which are aimed to construct a more physically-faithful and explainable data-driven forecast, and (3) Probabilistic-based, which account for the skillfulness of ensemble forecasts.


1. __Deterministic-based:__
    - [x] RMSE
    - [x] Bias
    - [x] Anomaly Correlation Coefficient (ACC)
    - [x] Multiscale Structural Similarity Index (MS-SSIM)
2. __Physics-based:__
    - [x] Spectral Divergence (SpecDiv)
    - [x] Spectral Residual (SpecRes)
    
3. __Probabilistic-based:__
    - [x] RMSE Ensemble
    - [x] Bias Ensemble
    - [x] ACC Ensemble
    - [x] MS-SSIM Ensemble
    - [x] SpecDiv Ensemble
    - [x] SpecRes Ensemble
    - [x] Continuous Ranked Probability Score (CRPS)
    - [x] Continuous Ranked Probability Skill Score (CRPSS)
    - [x] Spread
    - [x] Spread/Skill Ratio
    