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
from PIL import Image
from dataloader import Loader
from ResNet import ResNet18, BasicBlock
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

cfg = {
    'layer': '18',
    'batch_size': 32,
    'lr': 0.001,
    'weight_decay': 0.01,
    'epochs': 30,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
} 

def evaluate(model, loader_valid):
    # model.eval()
    correct=0
    total_loss=0
    Loss=nn.CrossEntropyLoss()
    with torch.no_grad():
        with tqdm(loader_valid, unit='batch', desc='Valid') as tqdm_loader:
            for index, (images,targets) in enumerate(tqdm_loader):  
                images,targets=images.to(cfg['device']),targets.to(cfg['device'],dtype=torch.long)
                predict=model(images)
                loss=Loss(predict,targets)
                total_loss+=loss.item()
                predict_class=predict.max(dim=1)[1]
                correct+=predict_class.eq(targets).sum().item()
                tqdm_loader.set_postfix(loss=loss.item(), avgloss=total_loss/(index+1), avgacc=correct/(cfg['batch_size']*(index+1)))
    avg_loss = total_loss / len(loader_valid.dataset)
    accuracy = correct / len(loader_valid.dataset)
    return avg_loss, accuracy

def test(model):
    model.eval()
    predict_result = list()
    with torch.no_grad():
        
        for data in loader_test:
            images = data.to(cfg['device'])
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predict_result.append(predicted.cpu())
        
        flat_result_list = [result for batch in predict_result for result in batch]
        save_result('./resnet_'+cfg['layer']+'_test.csv', flat_result_list)

def train(model, loader_train, loader_valid, layer):
    Loss=nn.CrossEntropyLoss()
    df=pd.DataFrame()
    df['epoch']=range(1,cfg['epochs']+1)
    best_model_wts=None
    best_evaluated_acc=0
    
    model.to(cfg['device'])
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=10, last_epoch=-1)
    acc_train=list()
    acc_valid=list()
    for epoch in range(1,cfg['epochs']+1):
        model.train()
        total_loss=0
        correct=0
        print(f'epcoh{epoch:>3d}')
        with tqdm(loader_train, unit='batch', desc='Train') as tqdm_loader:
            for index, (images,targets) in enumerate(tqdm_loader):
                images,targets=images.to(cfg['device']),targets.to(cfg['device'],dtype=torch.long)
                predict=model(images)
                loss=Loss(predict,targets)
                total_loss+=loss.item()
                correct+=predict.max(dim=1)[1].eq(targets).sum().item()
                optimizer.zero_grad()
                loss.backward()  # bp
                optimizer.step()
                tqdm_loader.set_postfix(loss=loss.item(), avgloss=total_loss/(index+1), avgacc=correct/(cfg['batch_size']*(index+1)))
            total_loss/=len(loader_train.dataset)
            acc=100.*correct/len(loader_train.dataset)
            acc_train.append(acc)
            valid_loss, valid_acc = evaluate(model, loader_valid)
            acc_valid.append(valid_acc)
        # update best_model_wts
        if valid_acc>best_evaluated_acc:
            best_evaluated_acc=valid_acc
            best_model_wts=copy.deepcopy(model.state_dict())
        scheduler.step()
    df['acc_train']=acc_train
    df['acc_valid']=acc_valid
    
    # save model
    torch.save(best_model_wts,os.path.join('resnet_'+cfg['layer']+'.pt'))
    
    return df
            

def save_result(csv_path, predict_result):
    df = pd.read_csv(csv_path)
    new_df = pd.DataFrame()
    new_df['ID'] = df['Path']
    new_df["label"] = predict_result
    new_df.to_csv("./312552047_resnet"+cfg['layer']+".csv", index=False)



if __name__ == "__main__":
    dataset = Loader(root = 'new_dataset', mode='train', layer=cfg['layer'])
    loader_train = DataLoader(dataset,batch_size=cfg['batch_size'],shuffle=True,num_workers=4)

    dataset = Loader(root = 'new_dataset', mode='valid', layer=cfg['layer'])
    loader_valid = DataLoader(dataset,batch_size=cfg['batch_size'],shuffle=False,num_workers=4)

    resnet = ResNet18(BasicBlock, [2,2,2,2]).to(device=cfg['device'])
    train(resnet, loader_train, loader_valid, layer=cfg['layer'])

    dataset = Loader(root = 'new_dataset', mode='test', layer=cfg['layer'])
    loader_test = DataLoader(dataset,batch_size=cfg['batch_size'],shuffle=False,num_workers=4)

    test(resnet)