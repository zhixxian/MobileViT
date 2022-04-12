# from operator import index
from __future__ import annotations
from sklearn.semi_supervised import LabelSpreading
import torch
import torchvision.transforms as transforms
import torchvision

from torch.utils.data import DataLoader,Dataset

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import PIL.Image

import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# img_path = '/HDD/zhixian/Cifar100/'
# annotation_path = '/HDD/zhixian/Cifar100/train.csv'
batch_size = 64

class Cifar100CustomDataset(Dataset):
    def __init__(self, img_path, annotation, transform):
        self.img_path = img_path
        self.annotation = pd.read_csv(annotation)
        # self.transform = transform
        # self.transform = transforms.Compose([transforms.ToTensor()])
        self.transform = torchvision.transforms.ToTensor()
        
        self.img_names = self.annotation[:]['image_id']
        self.labels = self.annotation[:]['fine_label_names']
        
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        # image = self.img_path + '/*.png'
        image = cv2.imread(self.img_path + self.img_names.iloc[idx])
        image = self.transform(image)
        labels = self.labels[idx]
        sample = {'image' : image, 'labels' : labels}
        
        return sample
     
train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((400, 400)),
                transforms.ToTensor()])

test_transform =transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((400, 400)),
                transforms.ToTensor()])

train_dataset=Cifar100CustomDataset(img_path='/HDD/zhixian/Cifar100/train_images',
                                    annotation='/HDD/zhixian/Cifar100/train.csv',
                                    transform=train_transform)
test_dataset=Cifar100CustomDataset(img_path='/HDD/zhixian/Cifar100/test_images',
                                    annotation='/HDD/zhixian/Cifar100/test.csv',
                                    transform=test_transform)

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=batch_size,
    shuffle=True
)

test_dataloader = DataLoader(
    test_dataset, 
    batch_size=4,
    shuffle=True
)