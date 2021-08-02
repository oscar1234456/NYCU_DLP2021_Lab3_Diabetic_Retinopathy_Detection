##import
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time
import os
import copy
from RetinopathyLoader import RetinopathyLoader,normalWeightGetter
import pickle
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
# Detect if we have a GPU available
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
##
model_name = "resnet50" # [resnet18, resnet50]

num_classes = 5

batch_size = 4

num_epochs = 10

learning_rate = 0.001

momentum_val = 0.9

weight_decay_val = 5e-4
# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract =False

usePretrained = False

##
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

##
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []
    train_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                print("---Training---")
            else:
                model.eval()   # Set model to evaluate mode
                print("---evaling---")

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for batch, (inputs, labels)in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if batch % 100 == 0:
                    print(f">>>batch [{batch+1}] loss:{loss.item()} ")
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print("Saved Model!")
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            if phase == 'train':
                train_acc_history.append(epoch_acc)

        print()
        # if phase=="val":
        #    scheduler.step(epoch_loss)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_acc_history, val_acc_history

##
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    # input_size = 0

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # model_ft.relu = nn.LeakyReLU()
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        # input_size = 224, ftrs = 512
    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        #input_size = 224, ftrs = 2048

    # return model_ft, input_size
    return model_ft

# Initialize the model for this run
# model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
model_ft = initialize_model(model_name, num_classes, feature_extract, use_pretrained=usePretrained)
# Print the model we just instantiated
print(model_ft)

##
print("Initializing Datasets and Dataloaders...")
train_data = RetinopathyLoader("./data", 'train')
test_data = RetinopathyLoader("./data", "test")
trainLoader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True) #, num_workers=4,pin_memory=True
testLoader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True) #, num_workers=4,pin_memory=True
dataloaders_dict = {"train": trainLoader, "val":testLoader}

##
# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=learning_rate, momentum=momentum_val, weight_decay=weight_decay_val)
#close weight_decay 0730 22:26
# optimizer_ft = optim.SGD(params_to_update, lr=learning_rate, momentum=momentum_val)
##
# classWeight = normalWeightGetter().to(device)
# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

## learning rate scheduler
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=2)

# Train and evaluate (return model, train_acc_history, test_acc_history)
model_ft, train_hist, test_hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft,num_epochs=num_epochs)

## Save my model
torch.save(model_ft.state_dict(), 'resnet50_nonpre_weight1.pth')

##Save Training & Testing Accuracy Result
with open('resnet50_nonpre_Training.pickle', 'wb') as f:
    pickle.dump(train_hist, f)
with open('resnet50_nonpre_Testing.pickle', 'wb') as f:
    pickle.dump(test_hist, f)