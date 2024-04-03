import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class ICLEVRDataset(Dataset):
    def __init__(self, args, mode='train', transforms=None):
        self.root = args.dataset_dir
        self.mode = mode
        self.transforms = transforms
        self.images, self.labels = self.get_data()

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transforms:
            image = self.transforms(image)
        
        return image, label

    def get_data(self):
        # 先讀 label 的對應檔, "gray cube" -> 0
        label_dict = json.load(open("objects.json"))
        data_dict = json.load(open(self.mode + ".json"))
        labels = list(data_dict.values())
        images = list(data_dict.keys())
        

        img_list, label_list = [], []
        for i in range(len(labels)):
            img_list.append("iclevr/" + images[i])

            onehot_label = np.zeros(24, dtype=np.float32)
            for j in range(len(labels[i])):
                onehot_label[label_dict[labels[i][j]]] = 1 
            label_list.append(onehot_label)

        return img_list, label_list