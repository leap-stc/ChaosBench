# Dataset Information

> __NOTE__: Hands-on exploration of the ChaosBench dataset in `notebooks/01a_s2s_data_exploration.ipynb`. All data has daily and 1.5-degree resolution.
 
1. __ERA5 Reanalysis__ for Surface-Atmosphere (1979-2023). The following table indicates the 48 variables (channels) that are available for Physics-based models. Note that the __Input__ ERA5 observations contains __ALL__ fields, including the unchecked boxes:

    Parameters/Levels (hPa) | 1000 | 925 | 850 | 700 | 500 | 300 | 200 | 100 | 50 | 10
    :---------------------- | :----| :---| :---| :---| :---| :---| :---| :---| :--| :-|
    Geopotential height, z ($gpm$) | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; |  
    Specific humidity, q ($kg kg^{-1}$) | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &nbsp; | &nbsp; | &nbsp; |  
    Temperature, t ($K$) | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; |  
    U component of wind, u ($ms^{-1}$) | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; |  
    V component of wind, v ($ms^{-1}$) | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; | &check; |  
    Vertical velocity, w ($Pas^{-1}$) | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &check; | &nbsp; | &nbsp; | &nbsp; | &nbsp; | &nbsp; |  
    
2. __LRA5 Reanalysis__ for Terrestrial (1979-2023)

| Acronyms    | Long Name                            | Units            |
|------------------|-------------------------------------------|-----------------------|
| asn     | snow albedo                               | (0 - 1)               |
| d2m     | 2-meter dewpoint temperature              | K                     |
| e       | total evaporation                         | m of water equivalent |
| es      | snow evaporation                          | m of water equivalent |
| evabs   | evaporation from bare soil                | m of water equivalent |
| evaow   | evaporation from open water               | m of water equivalent |
| evatc   | evaporation from top of canopy            | m of water equivalent |
| evavt   | evaporation from vegetation transpiration | m of water equivalent |
| fal     | forecaste albedo                          | (0 - 1)               |
| lai\_hv | leaf area index, high vegetation          | $m^2 m^{-2}$          |
| lai\_lv | leaf area index, low vegetation           | $m^2 m^{-2}$          |
| pev     | potential evaporation                     | m                     |
| ro      | runoff                                    | m                     |
| rsn     | snow density                              | $kg m^{-3}$           |
| sd      | snow depth                                | m of water equivalent |
| sde     | snow depth water equivalent               | m                     |
| sf      | snowfall                                  | m of water equivalent |
| skt     | skin temperature                          | K                     |
| slhf    | surface latent heat flux                  | $J m^{-2}$            |
| smlt    | snowmelt                                  | m of water equivalent |
| snowc   | snowcover                                 | \%                    |
| sp      | surface pressure                          | Pa                    |
| src     | skin reservoir content                    | m of water equivalent |
| sro     | surface runoff                            | m                     |
| sshf    | surface sensible heat flux                | $J m^{-2}$            |
| ssr     | net solar radiation                       | $J m^{-2}$            |
| ssrd    | download solar radiation                  | $J m^{-2}$            |
| ssro    | sub-surface runoff                        | m                     |
| stl1    | soil temperature level 1                  | K                     |
| stl2    | soil temperature level 2                  | K                     |
| stl3    | soil temperature level 3                  | K                     |
| stl4    | soil temperature level 4                  | K                     |
| str     | net thermal radiation                     | $J m^{-2}$            |
| strd    | downward thermal radiation                | $J m^{-2}$            |
| swvl1   | volumetric soil water layer 1             | $m^3 m^{-3}$          |
| swvl2   | volumetric soil water layer 2             | $m^3 m^{-3}$          |
| swvl3   | volumetric soil water layer 3             | $m^3 m^{-3}$          |
| swvl4   | volumetric soil water layer 4             | $m^3 m^{-3}$          |
| t2m     | 2-meter temperature                       | K                     |
| tp      | total precipitation                       | m                     |
| tsn     | temperature of snow layer                 | K                     |
| u10     | 10-meter u-wind                           | $ms^{-1}$             |
| v10     | 10-meter v-wind                           | $ms^{-1}$             |


3. __ORAS Reanalysis__ for Sea-Ice (1979-2023)

| Acronyms    | Long Name                            | Units            |
|------------------|-------------------------------------------|-----------------------|
| iicethic | sea ice thickness                          | m                            |
| iicevelu | sea ice zonal velocity                     | $ms^{-1}$  |
| iicevelv | sea ice meridional velocity                | $ms^{-1}$  |
| ileadfra | sea ice concentration                      | (0-1)      |
| so14chgt | depth of 14$^\circ$ isotherm               | m          |
| so17chgt | depth of 17$^\circ$ isotherm               | m          |
| so20chgt | depth of 20$^\circ$ isotherm               | m          |
| so26chgt | depth of 26$^\circ$ isotherm               | m          |
| so28chgt | depth of 28$^\circ$ isotherm               | m          |
| sohefldo | net downward heat flux                     | $W m^{-2}$ |
| sohtc300 | heat content at upper 300m  | $J m^{-2}$ |
| sohtc700 | heat content at upper 700m | $J m^{-2}$ |
| sohtcbtm | heat content for total water column        | $J m^{-2}$ |
| sometauy | meridonial wind stress                     | $N m^{-2}$ |
| somxl010 | mixed layer depth 0.01                     | m          |
| somxl030 | mixed layer depth 0.03                     | m          |
| sosaline | salinity                                   | Practical Salinity Units (PSU) |
| sossheig | sea surface height                         | m                       |
| sosstsst | sea surface temperature                    | $^\circ C$ |
| sowaflup | net upward water flux                      | $kg/m^2/s$ |
| sozotaux | zonal wind stress                          | $N m^{-2}$ |



