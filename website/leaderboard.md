# Leaderboard

We provide a leaderboard for both deterministic and probabilistic models/metrics.
This interactive page is inspired by the [WeatherBench 2 design](https://sites.research.google/weatherbench/).

We showcase select variables (`t-850`, `z-500`, `q-700`) on select metrics e.g., `RMSE`, `ACC` for deterministic models or `RMSE`, `CRPSS` for ensemble models. 
Learn more about the metrics [here](https://leap-stc.github.io/ChaosBench/metrics.html) and access the complete metrics across variables [here](https://huggingface.co/datasets/LEAP/ChaosBench/tree/main/logs). Unless stated, all results are evaluated for the year 2022.

## Deterministic Models
Deterministic models make a single (control) forecast. This method is particularly sensitive to initial condition, and is therefore not recommended for long-term chaotic prediction, such as in the S2S case. 

<iframe src="https://htmlpreview.github.io/?https://raw.githubusercontent.com/leap-stc/ChaosBench/main/website/html/control.html" width="100%" height="600px" frameborder="0"></iframe>

## Ensemble Models
Ensemble models make multiple forecasts (members) at a time to give us a collection of possible future outcomes. This (plus a well-dispersed ensemble) is particularly important for skillful S2S forecasting. 

<iframe src="https://htmlpreview.github.io/?https://raw.githubusercontent.com/leap-stc/ChaosBench/main/website/html/ensemble.html" width="100%" height="600px" frameborder="0"></iframe>
