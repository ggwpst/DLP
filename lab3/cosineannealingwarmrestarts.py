import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms, utils

cfg = {
    'layer': '18',
    'batch_size': 32,
    'lr': 0.001,
    'weight_decay': 0.03,
    'epochs': 105,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
} 
model = models.resnet18()
optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=2,eta_min=0)

fig = plt.figure()
x = list(range(100))
y = []
for epoch in range(100):
    scheduler.step()
    lr = scheduler.get_lr()
    y.append(scheduler.get_lr()[0])

plt.plot(x, y)
plt.title("CosineAnnealingWarmRestarts Learning Rate")
plt.xlabel("epoch")
plt.ylabel("Learning")
fig.savefig('CosineAnnealingWarmRestarts Learning Rate')