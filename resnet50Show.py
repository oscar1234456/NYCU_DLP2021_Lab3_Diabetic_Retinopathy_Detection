#Author: 310551076 Oscar Chen
#Course: NYCU DLP 2021 Summer
#Title: Lab3 Diabetic Retinopathy Detection
#Date: 2021/07/24
#Subject: Using ResNet18,50 Pretrained Model to classify Retina pic
#Email: oscarchen.cs10@nycu.edu.tw

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from RetinopathyLoader import RetinopathyLoaderRes18Test,RetinopathyLoaderRes50Test
from torchvision import datasets, models, transforms
from tester import test_model
from confusionMatrix import printConfusionMatrix
import warnings
warnings.filterwarnings('ignore')#For maxpool warning in pyTorch 1.9.0

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
if __name__ == '__main__':
    y_pred = list()
    y_ground = list()

    loss_fn = nn.CrossEntropyLoss()
    test_data = RetinopathyLoaderRes50Test("./data", "test")
    testLoader = DataLoader(test_data, batch_size=8, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    resnet50Model = models.resnet50(pretrained=False)
    set_parameter_requires_grad(resnet50Model, feature_extracting=False)
    num_ftrs = resnet50Model.fc.in_features
    resnet50Model.fc = nn.Linear(num_ftrs, 5)
    print(resnet50Model)
    print(">>load state....")
    pretrained_dict = torch.load('./testWeight/resnet50_weight1.pth')
    model_dict = resnet50Model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    resnet50Model.load_state_dict(pretrained_dict)
    # resnet50Model.load_state_dict(torch.load('./testWeight/resnet18_weight1.pth'))
    resnet50Model.to(device)

    dataloaders_dict = {"val": testLoader}
    print("ResNet50")
    test_model(resnet50Model, testLoader, loss_fn, device,y_pred,y_ground)
    # test_model_ori(resnet50Model, dataloaders_dict, criterion=loss_fn, device=device,num_epochs=10)

    printConfusionMatrix(y_pred, y_ground)