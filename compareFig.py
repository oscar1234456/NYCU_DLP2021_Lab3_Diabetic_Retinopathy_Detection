## importPhase
import matplotlib.pyplot as plt
import pickle
##resnet50 Loader
with open('./testWeight/resnet50_Testing.pickle', 'rb') as f:
    resnet50_Testing_gpu = pickle.load(f)
with open('./testWeight/resnet50_Training.pickle', 'rb') as f:
    resnet50_Training_gpu = pickle.load(f)

##resnet50 Preprocess
resnet50_Testing_cpu = list()
resnet50_Training_cpu = list()

for x in resnet50_Testing_gpu:
    resnet50_Testing_cpu.append(x.cpu())

for x in resnet50_Training_gpu:
    resnet50_Training_cpu.append(x.cpu())

## resnet50 Plot
epoch = 8
iter = [x+1 for x in range(epoch)]
# plt.xlim(-5,305)
# plt.ylim(64,102)
plt.plot(iter, resnet50_Training_cpu, 'r-', label="resnet50_Training(pre)")
plt.plot(iter, resnet50_Testing_cpu, 'b-', label="resnet50_Testing(pre)")

# plt.plot(iter, EEGnetELUTrain, 'm-', label="elu_train")
# plt.plot(iter, EEGnetELUTest, 'y-', label="elu_test")
plt.legend(loc='lower right')
plt.title("Result Comparison (ResNet50)")
plt.show()

##resnet18 Loader
with open('./testWeight/resnet18_Testing.pickle', 'rb') as f:
    resnet18_Testing_gpu = pickle.load(f)
with open('./testWeight/resnet18_Training.pickle', 'rb') as f:
    resnet18_Training_gpu = pickle.load(f)

##resnet18 Preprocess
resnet18_Testing_cpu = list()
resnet18_Training_cpu = list()

for x in resnet18_Testing_gpu:
    resnet18_Testing_cpu.append(x.cpu())

for x in resnet18_Training_gpu:
    resnet18_Training_cpu.append(x.cpu())

## resnet18 Plot
epoch = 10
iter = [x+1 for x in range(epoch)]
# plt.xlim(-5,305)
# plt.ylim(64,102)
plt.plot(iter, resnet18_Training_cpu, 'r-', label="resnet18_Training(pre)")
plt.plot(iter, resnet18_Testing_cpu, 'b-', label="resnet18_Testing(pre)")

# plt.plot(iter, EEGnetELUTrain, 'm-', label="elu_train")
# plt.plot(iter, EEGnetELUTest, 'y-', label="elu_test")
plt.legend(loc='lower right')
plt.title("Result Comparison (ResNet18)")
plt.show()