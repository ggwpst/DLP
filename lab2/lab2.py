import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataloader import read_bci_data

cfg = {
    'struct': 'deepconv',
    'batch_size': 1080,
    'lr': 0.001,
    'weight_decay': 0.01,
    'epochs': 300,
    'loss_function': nn.CrossEntropyLoss(),
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
} 
X_train,y_train,X_test,y_test=read_bci_data()
dataset=TensorDataset(torch.from_numpy(X_train),torch.from_numpy(y_train))
loader_train=DataLoader(dataset,batch_size=cfg['batch_size'],shuffle=True,num_workers=4)
dataset=TensorDataset(torch.from_numpy(X_test),torch.from_numpy(y_test))
loader_test=DataLoader(dataset,batch_size=cfg['batch_size'],shuffle=False,num_workers=4)

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

def train(loader_train,loader_test,activations,device):
    Loss=nn.CrossEntropyLoss()
    df=pd.DataFrame()
    df['epoch']=range(1,cfg['epochs']+1)
    best_model_wts={'ReLU':None,'LeakyReLU':None,'ELU':None}
    best_evaluated_acc={'ReLU':0,'LeakyReLU':0,'ELU':0}
    for name,activation in activations.items():
        if cfg['struct'] == 'eeg':
            model=EEGNet(activation)
        elif cfg['struct'] == 'deepconv':
            model=DeepConvNet(activation)
        model.to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        acc_train=list()
        acc_test=list()
        for epoch in range(1,cfg['epochs']+1):
            model.train()
            total_loss=0
            correct=0
            for idx,(data,target) in enumerate(loader_train):
                data=data.to(device,dtype=torch.float)
                target=target.to(device,dtype=torch.long) # target type has to be 'long'
                predict=model(data)
                loss=Loss(predict,target)
                total_loss+=loss.item()
                correct+=predict.max(dim=1)[1].eq(target).sum().item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss/=len(loader_train.dataset)
            correct=100.*correct/len(loader_train.dataset)
            if epoch%10==0:
                print(f'epcoh{epoch:>3d}  loss:{total_loss:.4f}  acc:{correct:.1f}%')
            acc_train.append(correct)
            model.eval()
            correct=evaluate(model,loader_test,device)
            acc_test.append(correct)
            if correct>best_evaluated_acc[name]:
                best_evaluated_acc[name]=correct
                best_model_wts[name]=copy.deepcopy(model.state_dict())
        df[name+'_train']=acc_train
        df[name+'_test']=acc_test

    return df,best_model_wts

def plot(dataframe):
    fig=plt.figure(figsize=(10,6))
    for name in dataframe.columns[1:]:
        plt.plot('epoch',name,data=dataframe)
    plt.legend()
    return fig



activations={'ReLU':nn.ReLU(),'LeakyReLU':nn.LeakyReLU(),'ELU':nn.ELU()}
if cfg['struct'] == 'eeg':
    df,best_model_wts=train(loader_train,loader_test,activations,cfg['device'])
    for name,model_wts in best_model_wts.items():
        torch.save(model_wts,os.path.join('eeg',name+'.pt'))
    figure=plot(df)
    figure.savefig('eeg result.png')
    for column in df.columns[1:]:
        print(f'{column} acc: {df[column].max()}')
elif cfg['struct'] == 'deepconv':
    df,best_model_wts=train(loader_train,loader_test,activations,cfg['device'])
    for name,model_wts in best_model_wts.items():
        torch.save(model_wts,os.path.join('deepconvnet',name+'.pt'))
    figure=plot(df)
    figure.savefig('deepconvnet result.png')
    for column in df.columns[1:]:
        print(f'{column} max acc: {df[column].max()}')