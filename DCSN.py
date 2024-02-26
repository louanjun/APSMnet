import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
import numpy as np
import scipy.io as sio
import os
import copy
import time
import csv
import argparse
import random
import math
apex = False
import spectral
import pandas as pd
from matplotlib import patches
class ConBNRelu3D(nn.Module):
    def __init__(self,in_channels=1, out_channels=24, kernel_size=(51, 3, 3), padding=0,stride=1):
        super(ConBNRelu3D,self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.padding=padding
        self.stride=stride
        self.conv=nn.Conv3d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride,padding=self.padding)
        self.bn=nn.BatchNorm3d(num_features=self.out_channels)
        self.relu = nn.ReLU(inplace=False)
    def forward(self,x):
        
        x = self.conv(x)
        x = self.bn(x)
        x= self.relu(x)
        return x
class ConBNRelu2D(nn.Module):
    def __init__(self,in_channels=1, out_channels=24, kernel_size=(51, 3, 3), stride=1,padding=0): 
        super(ConBNRelu2D,self).__init__()
        self.stride = stride
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.padding=padding
        self.conv=nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride,padding=self.padding)
        self.bn=nn.BatchNorm2d(num_features=self.out_channels)
        self.relu = nn.ReLU(inplace=False)
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x= self.relu(x)
        return x

class HyperCLR(nn.Module):
    def __init__(self,channel,output_units,windowSize):
        # 调用Module的初始化
        super(HyperCLR, self).__init__()
        self.channel=channel
        self.output_units=output_units
        self.windowSize=windowSize
        self.conv1 = ConBNRelu3D(in_channels=1,out_channels=8,kernel_size=(3,3,3),stride=1,padding=0)  #改kernel_size=(3,3,3)
        self.conv2 = ConBNRelu3D(in_channels=8,out_channels=16,kernel_size=(3,3,3),stride=1,padding=0)  #改kernel_size=(5,3,3)
        self.conv3 = ConBNRelu3D(in_channels=16,out_channels=32,kernel_size=(3,3,3),stride=1,padding=0)  #改kernel_size=(3,3,3)
        self.conv4 = ConBNRelu2D(in_channels=32*(self.channel-12), out_channels=64, kernel_size=(1, 1), stride=1, padding=0)  #改kernel_size=(3,3)
        self.pool=nn.AdaptiveAvgPool2d((2, 2))
        self.projector = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128,64),
        )
        self.fc=nn.Linear(256,128)
        self.relu1=nn.ReLU()
        self.fc2=nn.Linear(128, self.output_units)
    def forward(self, x):
        x = x.unsqueeze(1) #[50, 1, 9, 9, 9]
        x = self.conv1(x) #[50, 8, 7, 7, 7]
        x = self.conv2(x) #[50, 16, 5, 5, 5]
        x = self.conv3(x) #[50, 32, 3, 3, 3]
        x = x.reshape([x.shape[0],-1,x.shape[3],x.shape[4]]) #改[50, 96, 3, 3]
        x = self.conv4(x) #[50, 64, 3, 3] 
        x = self.pool(x) #[50, 64, 2, 2]
        x = x.reshape([x.shape[0], -1]) #[50, 256]
        h = self.projector(x) #[50, 64]
        x=self.fc(x) #[50, 128]
        x=self.relu1(x) #[50, 128]
        z=self.fc2(x) #[50, 6]
        return z #改h

def count_parameters(model):
    return HyperCLR(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    img = torch.randn(50, 9, 9, 9)

    net = HyperCLR(channel=15,output_units=6,windowSize=25)
    outputs  = net(img)
    print("outputs:",outputs.shape, count_parameters(net))

