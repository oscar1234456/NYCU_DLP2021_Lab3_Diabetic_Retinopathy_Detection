## importPhase
import matplotlib.pyplot as plt
import pickle
##resnet50 Loader
with open('./testWeight/resnet50_Testing.pickle', 'rb') as f:
    resnet50_Testing_gpu = pickle.load(f)
with open('./testWeight/resnet50_Training.pickle', 'rb') as f:
    resnet50_Training_gpu = pickle.load(f)
with open('./testWeight/resnet50_nonpre_Testing.pickle', 'rb') as f:
    resnet50_nonpre_Testing_gpu = pickle.load(f)
with open('./testWeight/resnet50_nonpre_Training.pickle', 'rb') as f:
    resnet50_nonpre_Training_gpu = pickle.load(f)

##resnet50 Preprocess
resnet50_Testing_cpu = list()
resnet50_Training_cpu = list()
resnet50_nonpre_Testing_cpu = list()
resnet50_nonpre_Training_cpu = list()

for x in resnet50_Testing_gpu:
    resnet50_Testing_cpu.append(x.cpu())

for x in resnet50_Training_gpu:
    resnet50_Training_cpu.append(x.cpu())

for x in resnet50_nonpre_Testing_gpu:
    resnet50_nonpre_Testing_cpu.append(x.cpu())

for x in resnet50_nonpre_Training_gpu:
    resnet50_nonpre_Training_cpu.append(x.cpu())

## resnet50 Plot
epoch = 8
iter = [x+1 for x in range(epoch)]
# plt.xlim(-5,305)
# plt.ylim(64,102)
plt.plot(iter, resnet50_Training_cpu, 'r-', label="Training(with pretraining)")
plt.plot(iter, resnet50_Testing_cpu, 'b-', label="Testing(with pretraining)")

plt.plot(iter, resnet50_nonpre_Training_cpu[:8], 'm-', label="Training(w/o pretraining)")
plt.plot(iter, resnet50_nonpre_Testing_cpu[:8], 'y-', label="Testing(w/o pretraining)")
plt.legend(loc='upper left')
plt.title("Result Comparison (ResNet50)")
plt.show()

##resnet18 Loader
with open('./testWeight/resnet18_Testing.pickle', 'rb') as f:
    resnet18_Testing_gpu = pickle.load(f)
with open('./testWeight/resnet18_Training.pickle', 'rb') as f:
    resnet18_Training_gpu = pickle.load(f)
with open('./testWeight/resnet18_nonpre_Testing.pickle', 'rb') as f:
    resnet18_nonpre_Testing_gpu = pickle.load(f)
with open('./testWeight/resnet18_nonpre_Training.pickle', 'rb') as f:
    resnet18_nonpre_Training_gpu = pickle.load(f)

##resnet18 Preprocess
resnet18_Testing_cpu = list()
resnet18_Training_cpu = list()
resnet18_nonpre_Testing_cpu = list()
resnet18_nonpre_Training_cpu = list()

for x in resnet18_Testing_gpu:
    resnet18_Testing_cpu.append(x.cpu())

for x in resnet18_Training_gpu:
    resnet18_Training_cpu.append(x.cpu())

for x in resnet18_nonpre_Testing_gpu:
    resnet18_nonpre_Testing_cpu.append(x.cpu())

for x in resnet18_nonpre_Training_gpu:
    resnet18_nonpre_Training_cpu.append(x.cpu())

## resnet18 Plot
epoch = 10
iter = [x+1 for x in range(epoch)]
# plt.xlim(-5,305)
# plt.ylim(64,102)
plt.plot(iter, resnet18_Training_cpu, 'r-', label="Training(with pretraining)")
plt.plot(iter, resnet18_Testing_cpu, 'b-', label="Testing(with pretraining)")

plt.plot(iter, resnet18_nonpre_Training_cpu, 'm-', label="Training(w/o pretraining)")
plt.plot(iter, resnet18_nonpre_Testing_cpu, 'y-', label="Testing(w/o pretraining)")
plt.legend(loc='upper left')
plt.title("Result Comparison (ResNet18)")
plt.show()