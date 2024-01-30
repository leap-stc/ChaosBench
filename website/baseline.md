# Baseline Models
We differentiate between physics-based and data-driven models. The former is succintly illustrated as in the figure below. 

## Model Definition
- __Physics-Based Models__:
    - [x] UKMO: UK Meteorological Office
    - [x] NCEP: National Centers for Environmental Prediction
    - [x] CMA: China Meteorological Administration
    - [x] ECMWF: European Centre for Medium-Range Weather Forecasts

- __Data-Driven Models__:
    - [x] Lagged-Autoencoder
    - [x] Fourier Neural Operator (FNO)
    - [x] ResNet
    - [x] UNet
    - [x] ViT/ClimaX
    - [x] PanguWeather
    - [x] Fourcastnetv2
    - [x] GraphCast

## Model Checkpoints
Checkpoints for data-driven models are accessible from [here](https://huggingface.co/datasets/LEAP/ChaosBench/tree/main/logs)

- Data-driven models are indicated by the `_s2s` suffix (e.g., `unet_s2s`). 

- The hyperparameter specifications are located in `version_xx/lightning_logs/hparams.yaml`. The hyperparameters encode the following:

    - `lead_time` (default: 1): arbitrary delta_t to finetune the model, for direct approach
    - `n_step` (default: 1): number of autoregressive step, s, for autoregressive approach
    - `only_headline`: if false, optimize for task 1; if true for task 2
    - `batch_size`: the batch size used for training
    - `train_years`: list of years used for training
    - `val_years`: list of years used for validation
    - `epochs`: number of epoch
    - `input_size`: number of input channel
    - `learning_rate`: update step at each iteration
    - `model_name`: the name of the model used for consistency
    - `num_workers`: number of workers used in dataloader
    - `output_size`: number of output channel
    - `t_max`: number of cosine learning rate scheduler cycle
    
__NOTE__: You will notice that for each data-driven model, there are 4 checkpoints. 

1. Version 0 - Task 1; autoregressive up to 1-day ahead
2. Version 1 - Task 1; autoregressive up to 5-day ahead
3. Version 2 - Task 2; autoregressive up to 1-day ahead
4. Version 3 - Task 2; autoregressive up to 5-day ahead

Only for `unet_s2s` do we have many more checkpoints. This is to check for the effect of `direct` vs. `autoregressive` training approach described in the paper. In particular, the `direct` models have the following version numbers,
1. Version {0, 4, 5, 6, 7, 8, 9, 10, 11, 12} - Task 1
2. Version {2, 13, 14, 15, 16, 17, 18, 19, 20, 21} - Task 2

Each element in the array corresponds to checkpoints optimized for each $\Delta T \in \{1, 5, 10, 15, 20, 25, 30, 35, 40, 44\}$.