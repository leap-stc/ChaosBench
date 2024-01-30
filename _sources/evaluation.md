# Evaluation

After training your model, you can simply perform evaluation by running:

1. __Autoregressive__
```
python eval_iter.py --model_name <YOUR_MODEL>_s2s --eval_years 2023 --version_num <VERSION_NUM>
```

2. __Direct__
```
python eval_direct.py --model_name <YOUR_MODEL>_s2s --eval_years 2023 --version_nums <VERSION_NUM> --task_num <TASK_NUM>
```

Where `<VERSION_NUM(S)>` corresponds to the version(s) that `pytorch_lightning` generated during training.

__For example__, in our `unet_s2s` baseline model, we can run:

- Autoregressive: `python eval_iter.py --model_name unet_s2s --eval_years 2023 --version_num 0`

- Direct: `python eval_direct.py --model_name unet_s2s --eval_years 2023 --version_nums 0 4 5 6 7 8 9 10 11 12 --task_num 1`


## Accessing Baseline Scores
You can access the complete scores (in `.csv` format) for data-driven, physics-based models, climatology, and persistence [here](https://huggingface.co/datasets/LEAP/ChaosBench/tree/main/logs). Below is a snippet from `logs/climatology/eval/rmse_climatology.csv`, where each row represents `<METRIC>`, such as `RMSE`, at each future timestep.

| z-10     | z-50     | z-100    | z-200    | z-300    | ... | w-1000   |
|----------|----------|----------|----------|----------|-----|----------|
| 539.7944 | 285.9499 | 215.14742| 186.43161| 166.28784| ... | 0.07912156|
| 538.9591 | 285.43832| 214.82317| 186.23743| 166.16902| ... | 0.07907272|
| 538.1366 | 284.96063| 214.51791| 186.04941| 166.04732| ... | 0.07903882|
| ...      | ...      | ...      | ...      | ...      | ... | ...      |
