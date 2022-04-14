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
    def __init__(self, img_path, annotation, transform = None):
        super().__init__()
        '''
        self.img_path = 이미지 폴더 경로
        self.annotation = 라벨 csv 파일 경로
        '''
        self.img_path = img_path
        self.annotation = pd.read_csv(annotation)
        self.transform = transform
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self,idx):
            
        img_name = os.path.join(self.img_path, str(self.annotation.iloc[idx]['image_id'])) # 이미지를 불러오기 위해 이미지 path와 jpg 이름 합쳐줌
        image = io.imread(img_name)
        label = self.annotation.iloc[idx]['fine_label_names'] # train.csv에서 라벨 부분만 갖고옴
        # label = np.asarray([label]) # 라벨 배열로 갖고오기
        # sample = {'image':image, 'label':label}
        
        # image = self.transform(sample['image'])
        # label = sample['label']
        labels_unique = set(label)
        keys = {key: value for key, value in zip(labels_unique, range(len(labels_unique)))}
        labels_onehot = torch.zeros(size=(len(label), len(keys)))
        for idx, label in enumerate(labels_onehot):
            labels_onehot[idx][keys[label]] = 1
        
        # if self.transform:
        #     image = self.transform(image)
            
        # return image, label
        # return self.transform(image), self.transform(label)
         
        return image, labels_onehot
        # return sample

train_transform = transforms.Compose([
                transforms.ToTensor()])

test_transform =transforms.Compose([
                transforms.ToTensor()])


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
