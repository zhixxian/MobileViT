# from operator import index

from skimage import io

import cv2
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
import os

batch_size = 64

class Cifar100CustomDataset(Dataset):
    def init(self, img_path, annotation, transform = None):
        '''
        self.img_path = 이미지 폴더 경로
        self.annotation = 라벨 csv 파일 경로
        '''
        self.img_path = img_path
        self.annotation = pd.read_csv(annotation)
        self.transform = transform
        
    def len(self):
        return len(self.annotation)
    
    def getitem(self,idx):
            
        img_name = os.path.join(self.img_path, str(self.annotation.iloc[idx]['image_id'])) # 이미지를 불러오기 위해 이미지 path와 jpg 이름 합쳐줌
        image = io.imread(img_name)
        label = self.annotation.iloc[:]['fine_label_names'] # train.csv에서 라벨 부분만 갖고옴
        label = np.array([label]) # 라벨 배열로 갖고오기

        sample = {'image':image, 'label':label}
        
        if self.transform:
            sample['image'] = self.transform(sample['image'])
         
        return sample['image'], sample['label']

train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((400, 400))])

test_transform =transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((400, 400))])


train_dataset=Cifar100CustomDataset(img_path='/HDD/zhixian/Cifar100/train_images/',
                                    annotation='/HDD/zhixian/Cifar100/train.csv',
                                    transform=train_transform)
test_dataset=Cifar100CustomDataset(img_path='/HDD/zhixian/Cifar100/test_images/',
                                    annotation='/HDD/zhixian/Cifar100/test.csv',
                                    transform=test_transform)

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=batch_size,
    shuffle=False,
)

test_dataloader = DataLoader(
    test_dataset, 
    batch_size=batch_size,
    shuffle=False
)

train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((400, 400))])

test_transform =transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((400, 400))])


train_dataset=Cifar100CustomDataset(img_path='/HDD/zhixian/Cifar100/train_images',
                                    annotation='/HDD/zhixian/Cifar100/train.csv',
                                    transform=train_transform)
test_dataset=Cifar100CustomDataset(img_path='/HDD/zhixian/Cifar100/test_images',
                                    annotation='/HDD/zhixian/Cifar100/test.csv',
                                    transform=test_transform)

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=batch_size,
    transform=train_transform,
    shuffle=False,
    # num_workers=4
)

test_dataloader = DataLoader(
    test_dataset, 
    batch_size=batch_size,
    transform=test_transform,
    shuffle=False
)
