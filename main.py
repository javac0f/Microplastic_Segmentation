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

from Datasets import ImageDataset

from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch
from sklearn.model_selection import train_test_split
import albumentations as album

# list of np.arrays representing image files


image_paths = []
mask_paths = []

# GET DATA PATHS
for filename in os.listdir(config.X_TRAIN_DATA):
    full_path = os.path.join(config.X_TRAIN_DATA, filename)
    image_paths.append(full_path)

# GET MASK PATHS
for filename in os.listdir(config.Y_TRAIN_DATA):
    full_path = os.path.join(config.Y_TRAIN_DATA, filename)
    mask_paths.append(full_path)

# TRAIN/VAL SPLIT
x_train, x_val, y_train, y_val = train_test_split(image_paths,
                                                  mask_paths, 
                                                  test_size=config.TEST_SPLIT, 
                                                  random_state = 52)


# MAKE DATASETS
train_dataset = ImageDataset(image_paths=x_train, mask_paths=y_train)
val_dataset = ImageDataset(image_paths=x_val, mask_paths=y_val)

# LOAD DATA
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=config.BATCH_SIZE,
                          shuffle=True,
                          num_workers=1)

valid_loader = DataLoader(dataset=val_dataset,
                          batch_size=config.BATCH_SIZE,
                          shuffle=False,
                          num_workers=1)

assert(train_loader)