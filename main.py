# DS LIBS
import numpy as np
import pandas as pd
import time

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
import utils

# DL LIBS
import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.utils.data import DataLoader

from Datasets import ImageDataset

from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch
from sklearn.model_selection import train_test_split
import albumentations as album



#%%

def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0

    model.train()
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader)
    return epoch_loss



def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(loader)
    return epoch_loss


#%%

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

# LOAD DATA AND VALIDATION DATASETS
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=config.BATCH_SIZE,
                          shuffle=True)

valid_loader = DataLoader(dataset=val_dataset,
                          batch_size=config.BATCH_SIZE,
                          shuffle=False)


# TRAIN THROUGH EPOCH CYCLES
best_valid_loss = float("inf")
for epoch in range(0,config.EPOCHS):

    # SETUP CHECKS    
    start_time = time.time()
    epoch_loss = 0.0

    train_loss = train(config.MODEL, 
                       train_loader, 
                       config.OPTIMIZER, 
                       config.LOSS,
                       config.DEVICE)
    
    valid_loss = evaluate(config.MODEL, 
                          valid_loader, 
                          config.LOSS, 
                          config.DEVICE)
    
    """ Saving the model """
    if valid_loss < best_valid_loss:
        data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {config.CHECKPOINT_PATH}"
        print(data_str)

        best_valid_loss = valid_loss
        torch.save(config.MODEL.state_dict(), config.CHECKPOINT_PATH)

    end_time = time.time()
    epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)

    data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
    data_str += f'\tTrain Loss: {train_loss:.3f}\n'
    data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
    print(data_str)


