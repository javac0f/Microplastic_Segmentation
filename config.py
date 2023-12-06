import torch
from JC_Models import JC_UNet
from segmentation_models_pytorch.utils.metrics import IoU
from segmentation_models_pytorch.utils.losses import DiceLoss

# FILE PREPROCESSING
X_TRAIN_DATA:str = "C:/Users/jcof2/Documents/Coding_Projects/Microplastic_Segmentation/DATA/train_data/Raw_Tiffs"
Y_TRAIN_DATA:str = "C:/Users/jcof2/Documents/Coding_Projects/Microplastic_Segmentation/DATA/train_data/Ground_Truth"
X_TEST_DATA:str = "C:/Users/jcof2/Documents/Coding_Projects/Microplastic_Segmentation/DATA/test_data/Raw_Tiffs"

TEST_SPLIT:int = 0.2

# MODEL HYPERPARAMS
EPOCHS:int = 12
BATCH_SIZE:int = 25
IMAGE_SIZE:int = 256
LEARNING_RATE:int = 1e-4


LOSS = DiceLoss()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
MODEL = JC_UNet().to(DEVICE)
CHECKPOINT_PATH:str = "DOCS/checkpoint.pth"

OPTIMIZER = torch.optim.Adam([dict(params=MODEL.parameters(), lr=LEARNING_RATE),])
SCHEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER, 'min', patience=5, verbose=True)
