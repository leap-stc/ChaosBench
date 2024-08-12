# Quickstart

> **_NOTE:_**  Only need the dataset? Jump directly to **Step 2**. You need ~150GB of free disk space. If you find any problems, feel free to contact us or raise a GitHub issue. 

**Step 0**: Clone the [ChaosBench](https://github.com/leap-stc/ChaosBench) Github repository

**Step 1**: Install package dependencies
```
$ cd ChaosBench
$ pip install -r requirements.txt
```

**Step 2**: Initialize the data space by running
```
$ cd data/
$ wget https://huggingface.co/datasets/LEAP/ChaosBench/resolve/main/process.sh
$ chmod +x process.sh
```
**Step 3**: Download the data 
```
# Required for inputs and climatology (e.g., normalization)
$ ./process.sh era5
$ ./process.sh lra5
$ ./process.sh oras5
$ ./process.sh climatology

# Optional: control (deterministic) forecasts
$ ./process.sh ukmo
$ ./process.sh ncep
$ ./process.sh cma
$ ./process.sh ecmwf

# Optional: perturbed (ensemble) forecasts
$ ./process.sh ukmo_ensemble
$ ./process.sh ncep_ensemble
$ ./process.sh cma_ensemble
$ ./process.sh ecmwf_ensemble
```
