# from operator import index
from __future__ import annotations
from sklearn.semi_supervised import LabelSpreading
import torch
import torchvision.transforms as transforms
import torchvision

from torch.utils.data import DataLoader,Dataset

import cv2
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch.multiprocessing
import torch
from torchvision.transforms import ToTensor
from torch.autograd import Variable

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

batch_size = 64

class Cifar100CustomDataset(Dataset):
    def __init__(self, img_path, annotation, transform=None):
        # self.img_path = 이미지 폴더 경로
        # self.anntation = 이미지 레이블 
      
        self.transform = transform
        self.img_path = img_path
        self.annotation = pd.read_csv(annotation)
        self.transform = transform

        self.image_list = glob.glob(self.img_path + '/*.jpg') # 이미지 가져오기
       
        self.img_names = self.annotation[:]['image_id'] # 이미지 이름
        self.label_names = self.annotation[:]['fine_label_names'] # 라벨 이름
        self.label_list = [0] * len(self.label_names) # # 라벨 이름 리스트화
    
    def __len__(self):
        return len(self.image_list) # 이미지개수 리턴
    
    def __getitem__(self, idx):
        self.img_path = self.image_list[idx]
        label = self.label_list[idx]
        img = ToTensor()(self.img_path).unsqueeze(0) # 여기서부터 55번째 줄 transform tensor 관련 오류로 인해 수정해본 부분입니다.
        img = torch.Tensor(img)
        img = Variable(img)
        img = transforms.ToPILImage()(img) # 여기까지!
            
        return img, label


train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Resize((400, 400))])

test_transform =transforms.Compose([
                transforms.ToTensor(),
                transforms.ToPILImage(),
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
