import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torchvision.transforms as transforms
import cv2
import torch

torch.manual_seed(2023)
np.random.seed(2023)

class ImageDataset(Dataset):
    def __init__(self, df, imgsz, added_size=0.25):
        self.df = df
        self.input_size = imgsz
        # self.img_dir = img_dir
        self.transform = transforms.ToTensor()
        # self.target_transform = target_transform
        self.added_size = added_size
        
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx].name
        image = cv2.imread(img_path)
        x1,y1,x2,y2 = self.df.iloc[idx].values[-5:-1]
        img_size = image.shape
        size = max(y2-y1,x2-x1)
        x2 = x1+size
        y2 = y1+size
        scale = size*self.added_size
        x1 = int(max(0,x1-scale))
        x2 = int(min(img_size[1],x1+size*(1+self.added_size)))
        y1 = int(max(0,y1-scale))
        y2 = int(min(img_size[0],y1+size*(1+self.added_size)))
        new_size = max(y2-y1,x2-x1)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[y1:y2,x1:x2]
        image = image.astype(np.float32)
        image = (image/255.0 - self.mean) / self.std
        # image = image.transpose([2, 0, 1])
        image = cv2.resize(image, self.input_size, interpolation = cv2.INTER_AREA)
        image = self.transform(image)
        
        labels = self.df.iloc[idx].values[:-5]
        labels = (labels - np.tile([x1,y1],68))/new_size
        labels = torch.Tensor(labels)
        
        meta = {'index': idx, 'new_size': new_size, 'left_corner_xy': (x1,y1)}
        
        return image, labels, meta
    