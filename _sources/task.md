# Task Overview

The `.yaml` file always has two sections: model_args and data_args. They allow you to change and modify the training processes to follow the configurations provided in the original manuscript.

```
model_args:
    model_name: <str>       # Name of your model e.g., 'unet_s2s'
    input_size: <int>       # Input size, default: 60 (ERA5)
    output_size: <int>      # Output size, default: 60 (ERA5)
    learning_rate: <float>  # Learning rate
    num_workers: <int>      # Number of workers
    epochs: <int>           # Number of epochs
    t_max: <int>            # Learning rate scheduler
    only_headline: <bool>   # Only optimized for config.HEADLINE_VARS
    
data_args:
    batch_size: <int>       # Batch size
    train_years: [...]      # Train years e.g., [1979, ...] 
    val_years: [...]        # Val years e.g., [2016, ...]
    n_step: <int, 1>        # Number of autoregressive training steps
    lead_time: <int, 1>     # N-day ahead forecast (for direct scheme)
    land_vars: [...]        # Extra LRA5 vars e.g., ['t2m', ...]
    ocean_vars: [...]       # Extra ORAS5 vars e.g., ['sosstsst', ...]
```

__Note__: 
1. If `only_headline = True`, then the model is optimized only for a subset of variables defined in `config.HEADLINE_VARS` (default: False).

2. If `n_step > 1`, the models will train over \( n \)-autoregressive steps (default: 1).

3. If `lead_time > 1`, the models will be able to forecast \( n \)-days ahead. For example, in our direct forecasts, if `lead_time = 4`, our model will predict the states 4 days into the future (default: 1).

4. If `land_vars` and/or `ocean_vars` are set with entries from the acronyms, these will be used as additional inputs and targets, on top of ERA5 variables (default: []).