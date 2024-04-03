import pandas as pd
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(10,6))
data18 = pd.read_csv('resnet18.csv')
data50 = pd.read_csv('resnet50.csv')
data152 = pd.read_csv('resnet152.csv')
plt.plot(data18['epoch'], data18['train_acc'], label="resnet18_train")
plt.plot(data18['epoch'], data18['valid_acc'], label="resnet18_valid")
plt.plot(data50['epoch'], data50['train_acc'], label="resnet50_train")
plt.plot(data50['epoch'], data50['valid_acc'], label="resnet50_valid")
plt.plot(data152['epoch'], data152['train_acc'], label="resnet152_train")
plt.plot(data152['epoch'], data152['valid_acc'], label="resnet152_valid")
plt.title("Accuracy Train vs Valid")
plt.legend(loc='best')
plt.xlabel("epoch")
plt.ylabel("Accuracy")
fig.savefig('Accuracy Train vs Valid.png')