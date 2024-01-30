# Task Overview

We presented __TWO__ task, where the model still takes as __inputs the FULL__ 60 variables, but the benchmarking __targets ALL or SUBSET__ of variable(s).

1. __Task 1️⃣: Full Dynamics Prediction.__
It is aimed at __ALL__ target channels simultaneously. This task is generally harder to perform but is useful to build a model that emulates the entire weather conditions.

2. __Task 2️⃣: Sparse Dynamics Prediction.__
It is aimed at a __SUBSET__ of target channel(s). This task is useful to build long-term forecasting model for specific variables, such as near-surface temperature (t-1000) or near-surface humidity (q-1000). 

__NOTE__: Before training your own model [instructions here](https://leap-stc.github.io/ChaosBench/training.html), you can specify the Task you are optimizing for by changing `only_headline` field in `chaosbench/configs/<YOUR_MODEL>_s2s.yaml` file:

- Task 1️⃣: `only_headline: False`

- Task 2️⃣: `only_headline: True`. By default, it is going to optimize on {t-850, z-500, q-700}. To change this, modify the `HEADLINE_VARS` field in `chaosbench/config.py` 

# Training Strategies
In addition, we also provide flags to train the model either __autoregressively__ or __directly__. 

- Autoregressive: Using current output as the next model input. The number of iterative steps is defined in the `n_step: <N_STEP>` field. For our baselines, we set `N_STEP = 5`.

- Direct: Directly targeting specific time in the future. The lead time can be specified in the `lead_time: <LEAD_TIME>` field. Ensure that `n_step: 1` for this case. For our baselines, we set `<LEAD_TIME>` $\in \{1, 5, 10, 15, 20, 25, 30, 35, 40, 44\}$