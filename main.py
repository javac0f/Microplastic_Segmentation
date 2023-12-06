# DS LIBS
import numpy as np
import pandas as pd

# VISUALIZATION LIBS
import cv2
from pprint import pprint
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

# SYSTEM
import os
import config
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# DL LIBS
import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.utils.data import DataLoader

from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch
from sklearn.model_selection import train_test_split
import albumentations as album





# GET DATA
train_data:list = [] # list of images loaded as numpy arrays with shape (2048px, 2048px, 3 channels)



# GET TRAIN DATA
image_dir = os.listdir(config.X_TRAIN_DATA)
mask_dir = os.listdir(config.Y_TRAIN_DATA)

# GET NUMBER OF SAMPLES
if(len(image_dir) == len(mask_dir)):
    num_samples = len(image_dir)


# GET LOAD IMAGES
for i in range(num_samples):
    image_path = os.path.join(config.X_TRAIN_DATA, image_dir[i])
    mask_path = os.path.join(config.Y_TRAIN_DATA, mask_dir[i])

    image = cv2.imread(image_path)

    

