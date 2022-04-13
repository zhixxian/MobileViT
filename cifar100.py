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

# img_path = '/HDD/zhixian/Cifar100/'
# annotation_path = '/HDD/zhixian/Cifar100/train.csv'
batch_size = 64

class Cifar100CustomDataset(Dataset):
    def __init__(self, img_path, annotation, transform=None):
        # self.to_tensor = torchvision.transforms.ToTensor()
        self.transform = transform
        self.img_path = img_path
        self.annotation = pd.read_csv(annotation)
        # self.transform = transforms.Compose([transforms.ToTensor()])
        #self.trans = torchvision.transforms.ToPILImage()      
        self.transform = transform

        self.image_list = glob.glob(self.img_path + '/*.jpg')  
        
        # self.Image_list = []
        # for self.img_path in self.image_list:
        #     self.Image_list.append(Image.open(self.img_path))        
            
        # self.img_names = self.annotation[:]['image_id'] # 이미지 이름
        self.label_names = self.annotation[:]['fine_label_names'] # 라벨 이름
        self.label_list = [0] * len(self.label_names)
    def __len__(self):
        return len(self.image_list) # 이미지개수
    
    def __getitem__(self, idx):
        # self.img_path = self.Image_list[idx]
        self.img_path = self.image_list[idx]
        label = self.label_list[idx]
        # img = Image.open(self.img_path)
        img = ToTensor()(self.img_path).unsqueeze(0)
        img = torch.Tensor(img)
        img = Variable(img)
        img = transforms.ToPILImage()(img)
        
        # if self.transform is not None:
        #     img = self.transform(img)
        # image = cv2.imread(self.img_names.iloc[idx])
        # image = cv2.imread(self.img_path)
        # img = self.transform(img)
        if self.transform is not None:
            image = self.transform(image)
        # labels = np.array([self.labels])

        # labels = self.labels[idx]
        # sample = {'image' : image, 'label' : labels}
        
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
    # transform=train_transform,
    shuffle=False,
    # num_workers=4
)

test_dataloader = DataLoader(
    test_dataset, 
    batch_size=batch_size,
    # transform=test_transform,
    shuffle=False
)
