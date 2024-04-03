import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import os
import copy
from torch.utils import data
import numpy as np
import random
import ttach as tta
from PIL import Image
from torch.utils import data
from dataloader import Loader
from ResNet import ResNet18, BasicBlock, Bottleneck, ResNet50, ResNet152
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

cfg = {
    'layer': '152',
    'batch_size': 32,
    'lr': 0.001,
    'weight_decay': 0.03,
    'epochs': 105,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
} 

def evaluate(model, loader_valid):
    confusion_matrix = np.zeros((2,2))
    model.eval()
    correct=0
    with torch.no_grad():
        for images, targets in loader_valid:
            images = images.to(cfg['device'])
            targets = targets.to(cfg['device'],dtype=torch.long)
            predict=model(images)
            predict_class=predict.max(dim=1)[1]
            correct+=predict_class.eq(targets).sum().item()
            for i in range(len(targets)):
                confusion_matrix[int(targets[i])][int(predict_class[i])]+=1
        correct = correct/len(loader_valid.dataset)
    confusion_matrix=confusion_matrix/confusion_matrix.sum(axis=1).reshape(2,1)
    return confusion_matrix, correct

def plot_confusion_matrix(confusion_matrix):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
    ax.xaxis.set_label_position('top')
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(i, j, '{:.2f}'.format(confusion_matrix[j, i]), va='center', ha='center')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    return fig

dataset = Loader(root = 'new_dataset', mode='valid', layer=cfg['layer'])
loader_valid = DataLoader(dataset,batch_size=cfg['batch_size'],shuffle=False,num_workers=4)
if cfg['layer'] == '18':
    resnet = ResNet18(BasicBlock, [2, 2, 2, 2]).to(device=cfg['device'])
    resnet.load_state_dict(torch.load('resnet_18.pt'))
    confusion_matrix, acc=evaluate(resnet, loader_valid)
    print(f'accuracy: {acc:.2f}')
    figure=plot_confusion_matrix(confusion_matrix)
    figure.savefig('ResNet18_confusion_matrix.png')
elif cfg['layer'] == '50':
    resnet = ResNet50(Bottleneck, [3, 4, 6, 3]).to(device=cfg['device'])
    resnet.load_state_dict(torch.load('resnet_50.pt'))
    confusion_matrix, acc=evaluate(resnet, loader_valid)
    print(f'accuracy: {acc:.2f}')
    figure=plot_confusion_matrix(confusion_matrix)
    figure.savefig('ResNet50_confusion_matrix.png')
elif cfg['layer'] == '152':
    resnet = ResNet152(Bottleneck, [3, 8, 36, 3]).to(device=cfg['device'])
    resnet.load_state_dict(torch.load('resnet_152.pt'))
    confusion_matrix, acc=evaluate(resnet, loader_valid)
    print(f'accuracy: {acc:.2f}')
    figure=plot_confusion_matrix(confusion_matrix)
    figure.savefig('ResNet152_confusion_matrix.png')