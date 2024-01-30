# Quickstart

**Step 1**: Clone the [ChaosBench](https://github.com/leap-stc/ChaosBench) Github repository

**Step 2**: Create local directory to store your data, e.g., 
```
cd ChaosBench
mkdir data
```

**Step 3**: Navigate to `chaosbench/config.py` and change the field `DATA_DIR = ChaosBench/data`

**Step 4**: Initialize the space by running
```
cd ChaosBench/data/
wget https://huggingface.co/datasets/juannat7/ChaosBench/blob/main/process.sh
chmod +x process.sh
```
**Step 5**: Download the data 

```
# NOTE: you can also run each line one at a time to retrieve individual dataset

./process.sh era5            # Required: For input ERA5 data
./process.sh climatology     # Required: For climatology
./process.sh ukmo            # Optional: For simulation from UKMO
./process.sh ncep            # Optional: For simulation from NCEP
./process.sh cma             # Optional: For simulation from CMA
./process.sh ecmwf           # Optional: For simulation from ECMWF
```
  