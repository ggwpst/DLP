import pandas as pd
import os
from torch.utils import data
import numpy as np
from PIL import Image
from torchvision import transforms

def getData(mode, layer):
    if mode == 'train':
        data = pd.read_csv('train.csv')
        path = data['Path']
        label = data['label']
        return path, label
    elif mode == 'valid':
        data = pd.read_csv('valid.csv')
        path = data['Path']
        label = data['label']
        return path, label
    elif mode == 'test':
        data = pd.read_csv('resnet_'+layer+'_test.csv')
        path = data['Path']
        return path


class Loader(data.Dataset):
    def __init__(self, mode, layer, root):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.mode = mode
        self.root = root
        if mode == 'test':
            self.img_name = getData(mode, layer)
        else:
            self.img_name, self.label = getData(mode,layer)
        self.data_len = len(self.img_name)
        print("> Found %d images..." % (len(self.img_name))) 
        self.transformations=transforms.Compose([transforms.CenterCrop(300), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),transforms.ToTensor(),
                                                transforms.Normalize((0.3749, 0.2602, 0.1857),(0.2526, 0.1780, 0.1291))])
        

    def __len__(self):
        """'return the size of dataset"""
        return self.data_len

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        if self.mode == 'test':
            img_path = os.path.join(self.root, self.img_name.iloc[index])
            image = Image.open(img_path)
            image = self.transformations(image)
            return image
        else:
            img_path = os.path.join(self.root, self.img_name.iloc[index])
            image = Image.open(img_path)
            image = self.transformations(image)
            label = int(self.label.iloc[index])
                
            return image, label