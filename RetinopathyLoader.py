import os

import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class RetinopathyLoader(Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        print(">Found %d images..." % (len(self.img_name)))
    def __len__(self):
       return  len(self.img_name)
    def __getitem__(self, index):
        transform1 = transforms.Compose(
            [
                transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            ]
        )
        img_path = self.root +'/' +self.img_name[index]+'.jpeg'
        image = Image.open(img_path).convert('RGB')
        label = self.label[index]
        imageConvert = transform1(image)
        # print("load index ", index)
        return imageConvert, label

def getData(mode):
    if mode == "train":
        img = pd.read_csv("./csv/train_img.csv")
        label = pd.read_csv("./csv/train_label.csv")
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv("./csv/test_img.csv")
        label = pd.read_csv("./csv/test_label.csv")
        return np.squeeze(img.values), np.squeeze(label.values)

if __name__ == '__main__':
        test_data = RetinopathyLoader("./data", 'test')
        test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
        print(test_dataloader)
        test_features, test_labels = next(iter(test_dataloader))
        print(f"Feature batch shape: {test_features.size()}")
        print(f"Labels batch shape: {test_labels.size()}")