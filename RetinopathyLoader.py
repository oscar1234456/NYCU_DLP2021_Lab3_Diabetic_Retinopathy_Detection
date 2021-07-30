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
        data_transform = {
            "train":transforms.Compose(
                [
                    # transforms.RandomRotation(degrees=(0,180)),
                    # transforms.RandomResizedCrop(224),
                    transforms.Resize(224),
                    # transforms.CenterCrop(224),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]
            ),
            "test":transforms.Compose(
                [
                    transforms.Resize(224),
                    # transforms.CenterCrop(224),
                    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]
            ),
        }
        # transform1 = transforms.Compose(
        #     [
        #         transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        #     ]
        # )
        img_path = self.root +'/' +self.img_name[index]+'.jpeg'
        image = Image.open(img_path).convert('RGB')
        label = self.label[index]
        imageConvert = data_transform[self.mode](image)
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
        train_data = RetinopathyLoader("./data", 'train')
        # test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
        # print(test_dataloader)
        # test_features, test_labels = next(iter(test_dataloader))
        # print(f"Feature batch shape: {test_features.size()}")
        # print(f"Labels batch shape: {test_labels.size()}")
        img, label = train_data[273]
        plt.figure()
        img_tran = img.numpy().transpose((1, 2, 0))  # [C,H,W]->[H,W,C]
        # plt.imshow((img_tran * 255).astype(np.uint8))
        plt.imshow(img_tran)
        plt.show()