import os
import cv2
import numpy as np


from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as album

import config



class ImageDataset(Dataset):
    
    def __init__(self, image_paths:str, mask_paths:str):
        self.images_path = image_paths
        self.masks_path = mask_paths
        self.n_samples = len(image_paths)


    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        
        # GET IMAGES
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = cv2.resize(image, (config.IMAGE_SIZE , config.IMAGE_SIZE))
        #image = self.transform(image)

        # GET MASK 
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_COLOR)
        mask = cv2.resize(mask, (config.IMAGE_SIZE , config.IMAGE_SIZE))
        #mask = self.transform(mask)

        # apply augmentations
        #if self.augmentation:
        #    sample = self.augmentation(image=image, mask=mask)
        #    image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        #if self.preprocessing:
        #    sample = self.preprocessing(image=image, mask=mask)
        #    image, mask = sample['image'], sample['mask']
            
        return image, mask
        

    def transform(self, image):
        img_mean, img_std = np.mean(image, axis=(0, 1)), np.std(image, axis=(0, 1))

        tensor_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean,std=img_std)
        ])
        
        return tensor_transforms(image)
    
