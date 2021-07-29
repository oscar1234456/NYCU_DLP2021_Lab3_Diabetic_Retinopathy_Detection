import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
class RetinopathyLoader(Dataset):
    def __init__(self, root, mode):
        pass
    def __len__(self):
        pass
    def __getitem__(self, index):
        pass


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
    a,b  = getData("test")
    c,d = getData("train")
