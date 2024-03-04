# Quickstart

**Step 1**: Clone the [ChaosBench](https://github.com/leap-stc/ChaosBench) Github repository

**Step 2**: Install package dependencies
```
cd ChaosBench
pip install -r requirements.txt
```

**Step 3**: Initialize the data space by running
```
cd data/
wget https://huggingface.co/datasets/LEAP/ChaosBench/resolve/main/process.sh
chmod +x process.sh
```
**Step 4**: Download the data 

```
# NOTE: you can also run each line one at a time to retrieve individual dataset

./process.sh era5            # Required: For input ERA5 data
./process.sh climatology     # Required: For climatology
./process.sh ukmo            # Optional: For simulation from UKMO
./process.sh ncep            # Optional: For simulation from NCEP
./process.sh cma             # Optional: For simulation from CMA
./process.sh ecmwf           # Optional: For simulation from ECMWF
```