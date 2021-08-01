import torch
from torch import nn

from torch.utils.data import DataLoader, TensorDataset
from RetinopathyLoader import RetinopathyLoaderRes18Test,RetinopathyLoaderRes50Test
from torchvision import datasets, models, transforms
from tester import test_model,test_model_ori
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
if __name__ == '__main__':
    loss_fn = nn.CrossEntropyLoss()
    test_data = RetinopathyLoaderRes18Test("./data", "test")
    testLoader = DataLoader(test_data, batch_size=8, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    resnet18Model = models.resnet18(pretrained=False)
    set_parameter_requires_grad(resnet18Model, feature_extracting=False)
    num_ftrs = resnet18Model.fc.in_features
    resnet18Model.fc = nn.Linear(num_ftrs, 5)
    print(resnet18Model)
    print(">>load state....")
    pretrained_dict = torch.load('./testWeight/resnet18_weight(we).pth')
    model_dict = resnet18Model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    resnet18Model.load_state_dict(pretrained_dict)
    # resnet18Model.load_state_dict(torch.load('./testWeight/resnet18_weight1.pth'))
    resnet18Model.to(device)

    dataloaders_dict = {"val": testLoader}
    print("ResNet18")
    test_model(resnet18Model, testLoader, loss_fn, device)
    # test_model_ori(resnet18Model, dataloaders_dict, criterion=loss_fn, device=device,num_epochs=10)

