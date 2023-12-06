import torch
from unet import UNet
from segmentation_models_pytorch.utils.metrics import IoU
from segmentation_models_pytorch.utils.losses import DiceLoss
import os
X_TRAIN_DATA:str = "C:/Users/jcof2/Documents/Coding_Projects/Microplastic_Segmentation/DATA/train_data/Raw_tiffs"
Y_TRAIN_DATA:str = "C:/Users/jcof2/Documents/Coding_Projects/Microplastic_Segmentation/DATA/train_data/Ground_Truth"
X_TEST_DATA:str = "C:/Users/jcof2/Documents/Coding_Projects/Microplastic_Segmentation/DATA/test_data/Raw_tiffs"


IMAGE_SIZE:int = 256
BATCH_SIZE:int = 25

TEST_SPLIT:int = 0.2

# Set flag to train the model or not. If set to 'False', only prediction is performed (using an older model checkpoint)
TRAINING = True

# Set num of epochs
EPOCHS = 12

# Set device: `cuda` or `cpu`
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define loss function
LOSS = DiceLoss()

# define metrics
METRICS = [
    IoU(threshold=0.5),
]

# DEFINE MODEL IN-USE
MODEL = UNet()

# DEFINE DEVICE USED
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define optimizer
OPTIMIZER = torch.optim.Adam([ 
    dict(params=MODEL.parameters(), lr=0.00008),
])

# define learning rate scheduler (not used in this NB)
LR_SCHEDULER = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    OPTIMIZER, T_0=1, T_mult=2, eta_min=5e-5,
)