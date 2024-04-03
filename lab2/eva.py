import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataloader import read_bci_data

cfg = {
    'struct': 'eeg',
    'batch_size': 1080,
    'lr': 0.001,
    'weight_decay': 0.01,
    'epochs': 300,
    'loss_function': nn.CrossEntropyLoss(),
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
} 

class EEGNet(nn.Module):
    def __init__(self,activation):
        super(EEGNet,self).__init__()
        self.firstconv=nn.Sequential(
            nn.Conv2d(1,16,kernel_size=(1,51),stride=(1,1),padding=(0,25),bias=False),
            nn.BatchNorm2d(16,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
        )
        self.depthwiseConv=nn.Sequential(
            nn.Conv2d(16,32,kernel_size=(2,1),stride=(1,1),groups=16,bias=False),
            nn.BatchNorm2d(32,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True),
            activation,
            nn.AvgPool2d(kernel_size=(1,4),stride=(1,4),padding=0),
            nn.Dropout(p=0.25)
        )
        self.seperableConv=nn.Sequential(
            nn.Conv2d(32,32,kernel_size=(1,15),stride=(1,1),padding=(0,7),bias=False),
            nn.BatchNorm2d(32,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True),
            activation,
            nn.AvgPool2d(kernel_size=(1,8),stride=(1,8),padding=0),
            nn.Dropout(p=0.25)
        )
        self.classify=nn.Linear(736,2)
    def forward(self,X):
        out=self.firstconv(X)
        out=self.depthwiseConv(out)
        out=self.seperableConv(out)
        out=out.view(out.shape[0],-1)
        out=self.classify(out)
        return out

class DeepConvNet(nn.Module):
    def __init__(self, activation):
        super(DeepConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5), padding='valid', bias=True),
            nn.Conv2d(25, 25, kernel_size=(2, 1), padding='valid', bias=True),
            nn.BatchNorm2d(25), 
            activation,
            nn.MaxPool2d(kernel_size=(1, 2)), 
            nn.Dropout(p=0.5)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5), padding='valid', bias=True),
            nn.BatchNorm2d(50), 
            activation,
            nn.MaxPool2d(kernel_size=(1, 2)), 
            nn.Dropout(p=0.5)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5), padding='valid', bias=True),
            nn.BatchNorm2d(100), 
            activation,
            nn.MaxPool2d(kernel_size=(1, 2)), 
            nn.Dropout(p=0.5)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5), padding='valid', bias=True),
            nn.BatchNorm2d(200), 
            activation,
            nn.MaxPool2d(kernel_size=(1, 2)), 
            nn.Dropout(p=0.5)
        )
        self.classify = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(in_features=8600, out_features=2, bias=True)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.classify(x)
        return x

def evaluate(model,loader_test,device):
    model.eval()
    correct=0
    for idx,(data,target) in enumerate(loader_test):
        data=data.to(device,dtype=torch.float)
        target=target.to(device,dtype=torch.long)
        predict=model(data)
        correct+=predict.max(dim=1)[1].eq(target).sum().item()
        
    correct=100.*correct/len(loader_test.dataset)
    return correct

model=EEGNet(nn.ReLU())
model.load_state_dict(torch.load(os.path.join('eeg','ReLU.pt')))
model.to(cfg['device'])
_,_,X_test,y_test=read_bci_data()
dataset=TensorDataset(torch.from_numpy(X_test),torch.from_numpy(y_test))
loader_test=DataLoader(dataset,batch_size=256,shuffle=False,num_workers=4)
acc=evaluate(model,loader_test,cfg['device'])
print(f'accuracy: {acc:.2f}%')