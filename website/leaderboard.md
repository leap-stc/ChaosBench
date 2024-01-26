# Leaderboard

We divide our metrics into 2 classes: (1) ML-based, which cover evaluation used in conventional computer vision and forecasting tasks, (2) Physics-based, which are aimed to construct a more physically-faithful and explainable data-driven forecast.

1. __Vision-based:__
    - [x] RMSE
    - [x] Bias
    - [x] Anomaly Correlation Coefficient (ACC)
    - [x] Multiscale Structural Similarity Index (MS-SSIM)
2. __Physics-based:__
    - [x] Spectral Divergence (SpecDiv)
    - [x] Spectral Residual (SpecRes)
    

For all models (data-driven, physics-based, etc), there is a folder named `eval/`. This contains individual `.csv` files for each metric (e.g., SpecDiv, RMSE). Within each file, it contains scores for all channels in question (e.g., the entire 60 for task 1, arbitrary n for task 2, or 48 for physics-based models) across 44-day lead time.