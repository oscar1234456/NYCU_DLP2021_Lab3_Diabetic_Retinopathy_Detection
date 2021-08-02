#Author: 310551076 Oscar Chen
#Course: NYCU DLP 2021 Summer
#Title: Lab3 Diabetic Retinopathy Detection
#Date: 2021/07/24
#Subject: Using ResNet18,50 Pretrained Model to classify Retina pic
#Email: oscarchen.cs10@nycu.edu.tw

##import process detect CUDA
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import time
import copy
from RetinopathyLoader import RetinopathyLoader,normalWeightGetter
import pickle
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
## Hyperparameter
model_name = "resnet18" # [resnet18, resnet50] choose the model

num_classes = 5

batch_size = 5

num_epochs = 20

learning_rate = 0.1

momentum_val = 0.9

weight_decay_val = 5e-4

feature_extract =False #True: Only train the last layer

usePretrained = True #True:Using Pretrained weight

## Depend on feature_extract, close the requires_grad fo model param
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

## train_model Function
def train_model(model, dataloaders, criterion, optimizer,scheduler ,num_epochs=10):
    since = time.time() #count execute time

    val_acc_history = [] #record val acc
    train_acc_history = [] #record train acc

    best_model_weight = copy.deepcopy(model.state_dict()) #record the weight of best model
    best_acc = 0.0 #record the best model acc

    for epoch in range(num_epochs):
        print(optimizer.param_groups[0]['lr'])
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('------------')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                print("---Training---")
            else:
                model.eval()
                print("---evaluating---")

            now_loss = 0.0
            now_corrects = 0

            for batch, (inputs, labels)in enumerate(dataloaders[phase]):
                # size = len(dataloaders[phase])
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                now_loss += loss.item() * inputs.size(0)
                now_corrects += torch.sum(preds == labels.data)
                if batch % 100 == 0:
                    print(f">>>batch [{batch+1}] loss:{loss.item()} ") #print now batch status
            size = len(dataloaders[phase].dataset)
            epoch_loss = now_loss / size
            epoch_acc = now_corrects.double() / size

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weight = copy.deepcopy(model.state_dict()) #record mbest model weight
                print("Saved Model!")
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            if phase == 'train':
                train_acc_history.append(epoch_acc)

        print()
        scheduler.step(epoch_acc) # use for adaptive learning rate control (Experiment Phase)

    time_elapsed = time.time() - since #time end
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_weight)
    return model, train_acc_history, val_acc_history

## create the model architecture from TorchVision
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    preModel = None

    if model_name == "resnet18":
        preModel = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(preModel, feature_extract)
        # model_ft.relu = nn.LeakyReLU() #[Experiment Phase]
        num_features = preModel.fc.in_features
        preModel.fc = nn.Linear(num_features, num_classes) #Add Fully-Connected Layer
        # input = 224, ftrs = 512
    elif model_name == "resnet50":
        preModel = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(preModel, feature_extract)
        num_features = preModel.fc.in_features
        preModel.fc = nn.Linear(num_features, num_classes) #Add Fully-Connected Layer
        #input 224, ftrs = 2048

    return preModel

model_ft = initialize_model(model_name, num_classes, feature_extract, use_pretrained=usePretrained)
print(model_ft)

## DataLoader
print("Initializing Datasets and Dataloaders...")
train_data = RetinopathyLoader("./data", 'train')
test_data = RetinopathyLoader("./data", "test")
trainLoader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True)
testLoader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True)
dataloaders_dict = {"train": trainLoader, "val":testLoader}

##
model_ft = model_ft.to(device)

params_to_update = model_ft.parameters() #load model all parameters
print("Params to learn:")
if feature_extract:
    # all param will not be trained (except last layer)
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    # all param will be trained
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

optimizer_ft = optim.SGD(params_to_update, lr=learning_rate, momentum=momentum_val, weight_decay=weight_decay_val)
# close weight_decay[Experiment Phase]:
# optimizer_ft = optim.SGD(params_to_update, lr=learning_rate, momentum=momentum_val)
##
# classWeight = normalWeightGetter().to(device) #For Imbalanced Data Approach Testing
# criterion = nn.CrossEntropyLoss(classWeight)
criterion = nn.CrossEntropyLoss()

## learning rate scheduler [Experiment Phase]
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft,factor=0.1, patience=2,mode='max',verbose=True)

model_ft, train_hist, test_hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft,scheduler=scheduler,num_epochs=num_epochs)

## Save Best model
torch.save(model_ft.state_dict(), 'resnet18_weight1.pth')

##Save Training & Testing Accuracy Result
with open('resnet18_Training.pickle', 'wb') as f:
    pickle.dump(train_hist, f)
with open('resnet18_Testing.pickle', 'wb') as f:
    pickle.dump(test_hist, f)