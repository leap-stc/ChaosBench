# Training

> __NOTE__: Hands-on modeling and training workflow can be found in `notebooks/02a_s2s_modeling.ipynb` and `notebooks/03a_s2s_train.ipynb`

We will outline how one can implement their own data-driven models. Several examples, including ED, ResNet, UNet, and FNO have been provided in the main repository. 

**Step 1**: Define your model class.

```
# An example can be found for e.g. <YOUR_MODEL> == fno

$ touch chaosbench/models/<YOUR_MODEL>.py
```

**Step 2**: Import and initialize your model in the main `chaosbench/models/model.py` file, given the pseudocode below.

```
# Examples for lagged_ae, fno, resnet, unet are provided

import lightning.pytorch as pl
from chaosbench.models import YOUR_MODEL

class S2SBenchmarkModel(pl.LightningModule):

    def __init__(
        self, 
        ...
    ):
        super(S2SBenchmarkModel, self).__init__()
        
        # Initialize your model
        self.model = YOUR_MODEL.BEST_MODEL(...)

        # The rest of model construction logic
      
```

**Step 3**: Run the `train.py` script. We recommend using GPUs for training.

```
# The _s2s suffix identifies data-driven models

$ python train.py --config_filepath chaosbench/configs/<YOUR_MODEL>_s2s.yaml
```

> __NOTE__: Now you will notice that there is a `.yaml` file. We will define the definition of each field next, allowing for greater control over different training strategies.
