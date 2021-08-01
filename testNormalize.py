import torch
from RetinopathyLoader import RetinopathyLoader
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def main():

    data_set =  train_data = RetinopathyLoader("./data", 'train')
    data_loader = DataLoader(data_set, batch_size=8, num_workers=4, shuffle=False)

    nb_samples = 0.
    channel_mean = torch.zeros(3)
    channel_std = torch.zeros(3)
    for images, targets in data_loader:
        # scale image to be between 0 and 1
        N, C, H, W = images.shape[:4]
        data = images.view(N, C, -1)

        channel_mean += data.mean(2).sum(0)
        channel_std += data.std(2).sum(0)
        nb_samples += N

    channel_mean /= nb_samples
    channel_std /= nb_samples
    print(channel_mean, channel_std)


if __name__ == '__main__':
    main()