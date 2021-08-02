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

def test_model_ori(model, dataloaders, criterion,device, num_epochs=25):

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['val']:
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
                # optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        # optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if batch % 100 == 0:
                    print(f">>>batch [{batch+1}] loss:{loss.item()} ")
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))


        print()

    print('Best val Acc: {:4f}'.format(epoch_acc))

