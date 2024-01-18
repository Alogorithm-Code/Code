#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F



class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(28*28,512)
        self.fc2 = nn.Linear(512,512)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(512,10)

    def forward(self,x):
        x = x.view(-1,28*28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        feature = self.relu(x)

        x = self.fc3(feature)

        return x, feature
        


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2048, 200)
        self.fc2 = nn.Linear(200, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        #print(x.size())
        x = self.fc1(x)
        feature = F.relu(x)
        x = self.fc2(feature)
        return x, feature





###Conditional generator
def one_hot_embedding(labels, num_classes=10):
    y = torch.eye(num_classes)
    return y[labels].view(-1, num_classes)



class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3, img_size=28, num_classes=10):
        super(Generator, self).__init__()

        self.init_size = img_size//4
        self.l1 = nn.Sequential(nn.Linear(nz+num_classes, ngf*2*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(ngf*2),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(nc, affine=False) 
        )

    def forward(self, z, label):
        z_c = torch.cat((z, label), dim=1)
        #z_c = torch.cat((z, label), dim=1)
        out = self.l1(z_c.view(z_c.shape[0],-1))
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img
