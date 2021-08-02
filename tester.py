#Author: 310551076 Oscar Chen
#Course: NYCU DLP 2021 Summer
#Title: Lab3 Diabetic Retinopathy Detection
#Date: 2021/07/24
#Subject: Using ResNet18,50 Pretrained Model to classify Retina pic
#Email: oscarchen.cs10@nycu.edu.tw

import torch


def test_model(model, testloaders, criterion, device, y_pred, y_ground):
    print("-----Testing Start(Testing Set)-----")

    model.eval()
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in testloaders:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            y_pred.extend(preds.view(-1).detach().cpu().numpy())  # 將preds預測結果detach出來，並轉成numpy格式
            y_ground.extend(labels.view(-1).detach().cpu().numpy())

    all_loss = running_loss / len(testloaders.dataset)
    all_acc = running_corrects.double() / len(testloaders.dataset)

    print('Acc: {:.0f}%'.format(all_acc*100))
    return y_pred, y_ground
