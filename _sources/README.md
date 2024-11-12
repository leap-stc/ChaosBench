# ChaosBench: A Multi-Channel, Physics-Based Benchmark for Subseasonal-to-Seasonal Climate Prediction

<div align="center">
<a href="https://leap-stc.github.io/ChaosBench"><img src="https://img.shields.io/badge/View-Documentation-blue?style=for-the-badge)" alt="Homepage"/></a>
  <a href="https://arxiv.org/abs/2402.00712"><img src="https://img.shields.io/badge/ArXiV-2402.00712-b31b1b.svg" alt="arXiv"/></a>
<a href="https://huggingface.co/datasets/LEAP/ChaosBench"><img src="https://img.shields.io/badge/Dataset-HuggingFace-ffd21e" alt="Huggingface Dataset"/></a>
<a href="https://github.com/leap-stc/ChaosBench/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-GNU%20GPL-green" alt="License Badge"/></a>
</div>
</br>

ChaosBench is a benchmark project to improve and extend the predictability range of deep weather emulators to the subseasonal-to-seasonal (S2S) range. Predictability at this scale is more challenging due to its: (1) __double sensitivities__ to intial condition (in weather-scale) and boundary condition (in climate-scale), (2) __butterfly effect__, and our (3) __inherent lack of understanding__ of physical processes operating at this scale. Thus, given the __high socioeconomic stakes__ for accurate, reliable, and stable S2S forecasts (e.g., for disaster/extremes preparedness), this benchmark is timely for DL-accelerated solutions.



## Features

![Overview of ChaosBench](docs/scheme/chaosbench_scheme-scheme.jpg)

1️⃣ __Diverse Observations__. Spanning over 45 years (1979-), we include ERA5/LRA5/ORAS5 reanalysis for a fully-coupled Earth system emulation (atmosphere-terrestrial-sea-ice)

2️⃣ __Diverse Baselines__. Wide selection of physics-based forecasts from leading national weather agencies in Europe, the UK, America, and Asia

3️⃣ __Differentiable Physics Metrics__. In addition to deterministic and probabilistic metrics, we introduce two differentiable physics-based metrics to minimize the decay of power spectra at long forecasting horizon (reduce blurriness)

4️⃣ __Large-Scale Benchmarking__. Systematic large-scale evaluation for state-of-the-art ML-based weather emulators like ViT/ClimaX, PanguWeather, GraphCast, and FourcastNetV2

## Getting Started
- [Motivation](https://leap-stc.github.io/ChaosBench/motivation.html)
- [Quickstart](https://leap-stc.github.io/ChaosBench/quickstart.html)
- [Dataset Overview](https://leap-stc.github.io/ChaosBench/dataset.html)


## Build Your Own Model
- [Training](https://leap-stc.github.io/ChaosBench/training.html)
- [Evaluation](https://leap-stc.github.io/ChaosBench/evaluation.html)

## Benchmarking
- [Baseline Models](https://leap-stc.github.io/ChaosBench/baseline.html)
- [Leaderboard](https://leap-stc.github.io/ChaosBench/leaderboard.html)

## Citation
If you find any of the code and dataset useful, feel free to acknowledge our work through:

```bibtex
@article{nathaniel2024chaosbench,
  title={Chaosbench: A multi-channel, physics-based benchmark for subseasonal-to-seasonal climate prediction},
  author={Nathaniel, Juan and Qu, Yongquan and Nguyen, Tung and Yu, Sungduk and Busecke, Julius and Grover, Aditya and Gentine, Pierre},
  journal={arXiv preprint arXiv:2402.00712},
  year={2024}
}

