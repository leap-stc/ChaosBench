# Task Overview

We presented __TWO__ task, where the model still takes as __inputs the FULL__ 60 variables, but the benchmarking __targets ALL or SUBSET__ of variable(s).

1. __Task 1️⃣: Full Dynamics Prediction.__
It is aimed at __ALL__ target channels simultaneously. This task is generally harder to perform but is useful to build a model that emulates the entire weather conditions.

2. __Task 2️⃣: Sparse Dynamics Prediction.__
It is aimed at a __SUBSET__ of target channel(s). This task is useful to build long-term forecasting model for specific variables, such as near-surface temperature (t-1000) or near-surface humidity (q-1000). 

    - __NOTE:__ By default, it is going to optimize on {t-850, z-500, q-700}. To change this, modify the `HEADLINE_VARS` field in `chaosbench/config.py` 
    
# Training Strategies
In addition, we also provide flexibility to train models in either __autoregressive__ or __direct__ manner.

1. __Strategy 1️⃣: Autoregressive.__ Using current output as the next model input.

2. __Strategy 2️⃣: Direct.__ Directly targeting specific time in the future. 


The [next section](https://leap-stc.github.io/ChaosBench/training.html) is going to discuss how to initialize your model and perform training depending on the task and strategy you choose.