# ChaosBench: A Multi-Channel, Physics-Based Benchmark for Subseasonal-to-Seasonal Climate Prediction


ChaosBench is a benchmark project to improve long-term forecasting of chaotic systems, in particular subseasonal-to-seasonal (S2S) climate, using ML approaches.

🌐: https://leap-stc.github.io/ChaosBench

📚: https://arxiv.org/abs/2402.00712

🤗: https://huggingface.co/datasets/LEAP/ChaosBench

## Features

![Overview of ChaosBench](docs/scheme/chaosbench_scheme-scheme.jpg)

1️⃣ __Diverse Observations__. Spanning over 45 years (1979 - 2023), we include ERA5/LRA5/ORAS5 reanalysis for a fully-coupled Earth system emulation (atmosphere-terrestrial-sea-ice)

2️⃣ __Diverse Baselines__. Wide selection of physics-based forecasts from leading national agencies in Europe, the UK, America, and Asia

3️⃣ __Differentiable Physics Metrics__. Introduces two differentiable physics-based metrics to minimize the decay of power spectra at long forecasting horizon (blurriness)

4️⃣ __Large-Scale Benchmarking__. Systematic evaluation (deterministic, probabilistic, physics-based) for state-of-the-art ML-based weather emulators like ViT/ClimaX, PanguWeather, GraphCast, and FourcastNetV2

## Getting Started
- [Motivation](https://leap-stc.github.io/ChaosBench/motivation.html)
- [Quickstart](https://leap-stc.github.io/ChaosBench/quickstart.html)
- [Dataset Overview](https://leap-stc.github.io/ChaosBench/dataset.html)


## Build Your Own Model
- [Training](https://leap-stc.github.io/ChaosBench/training.html)
- [Evaluation](https://leap-stc.github.io/ChaosBench/evaluation.html)

## Benchmarking
- [Baseline Models](https://leap-stc.github.io/ChaosBench/baseline.html)
