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
x:list = [] # list of images loaded as numpy arrays with shape (2048px, 2048px, 3 channels)
y:list = [] # list of images loaded as numpy arrays with shape (2048px, 2048px, 3 channels)

# GET X DATA
for filename in os.listdir(config.X_TRAIN_DATA):
    full_path = os.path.join(config.X_TRAIN_DATA, filename)
    image = cv2.imread(full_path, cv2.IMREAD_COLOR)
    x.append(image)

# GET Y DATA
for filename in os.listdir(config.Y_TRAIN_DATA):
    full_path = os.path.join(config.Y_TRAIN_DATA, filename)
    mask = cv2.imread(full_path, cv2.IMREAD_COLOR)
    y.append(mask)


# TRANSFORM IMAGES 








## SPLIT DATA INTO TRAIN AND VAL
#x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=config.TEST_SPLIT, random_state=52)