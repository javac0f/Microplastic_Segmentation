import cv2
from torch.utils.data import Dataset
import numpy as np
import config
from torchvision import transforms
import albumentations as album

class ImageDataset(Dataset):
    
    def __init__(self, image_list:str, mask_list:str, augmentation = None, preprocessing = None):
        self.image_list = image_list
        self.mask_list = mask_list
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.image_list)


    def __getitem__(self, index):
        
        # GET ITEM
        image = cv2.resize(self.image_list[index], (config.IMAGE_SIZE , config.IMAGE_SIZE))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)

        # GET MASK 
        mask = cv2.resize(self.mask_list[index], (config.IMAGE_SIZE , config.IMAGE_SIZE))
        #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = self.transform(mask)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        

    def transform(self, image):
        img_mean, img_std = np.mean(image, axis=(0, 1)), np.std(image, axis=(0, 1))

        tensor_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean,std=img_std)
        ])
        
        return tensor_transforms(image)
    

class ImageTransforms():
        def get_training_augmentation():
            train_transform = [    
                album.RandomCrop(height=256, width=256, always_apply=True),
                album.OneOf(
                    [
                        album.HorizontalFlip(p=1),
                        album.VerticalFlip(p=1),
                        album.RandomRotate90(p=1),
                    ],
                    p=0.75,
                ),
            ]
            return album.Compose(train_transform)


        def get_validation_augmentation():   
            # Add sufficient padding to ensure image is divisible by 32
            test_transform = [
                album.PadIfNeeded(min_height=1536, min_width=1536, always_apply=True, border_mode=0),
            ]
            return album.Compose(test_transform)


        def to_tensor(self,x):
            return x.transpose(2, 0, 1).astype('float32')


        def get_preprocessing(self,preprocessing_fn=None):
            """Construct preprocessing transform    
            Args:
                preprocessing_fn (callable): data normalization function 
                    (can be specific for each pretrained neural network)
            Return:
                transform: albumentations.Compose
            """   
            _transform = []
            if preprocessing_fn:
                _transform.append(album.Lambda(image=preprocessing_fn))
            _transform.append(album.Lambda(image=self.to_tensor, mask=self.to_tensor))
                
            return album.Compose(_transform)