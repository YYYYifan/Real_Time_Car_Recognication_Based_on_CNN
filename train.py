from packages import prepare
from packages import dataset
from packages import models

from torch.utils.data import DataLoader
import torch
import datetime
import json


epoch_size = 40
learnRate = 0.001
batch_size = 12

with open("./parameter.json", 'r') as file_obj:
    parameter = json.load(file_obj)
    len_each_subset_in_train = parameter["len_each_subset_in_train"]    


myImage = prepare.imagePocess(save=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(device))
# Load datas to torchvision.dataset .
train_dataset = dataset.Mydataset(myImage.train, len_each_subset_in_train)
# Load dataset to torchvision.dataloder .
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Load Convorlotion Neural Network Models.
net = models.MobileNetV2(2)    
net.to(device)

# Define optimizer and loss func.
optimizer = torch.optim.SGD(net.parameters(), lr=learnRate, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()

# Adjust learning rate by each epoch
def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


print("Start Train")
with open("./result/train.log", "w") as log_obj:
    start = datetime.datetime.now()    
    index_size = 0
    for time in range(epoch_size):
        adjust_learning_rate(optimizer,time,learnRate)
        total = 0
        correct = 0
        net.train(True)
        for index, (image,labels) in enumerate(train_dataloader):
                image = image.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                # print("input shape",image.shape)
                outputs = net(image)
                # print(outputs.shape,labels.shape)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()                
                _, predicted = outputs.max(1)
                total += outputs.size(0)
                correct += predicted.eq(labels).sum().item()
                Accuracy = correct/total
                print("Epoch: {}/{}, idx:{} Loss:{:.4f} Accuracy:{:.4f}".format(time+1,epoch_size, index, loss,Accuracy))
                logData = [time, index, loss.item(), Accuracy]
                log_obj.write(str(logData))
                log_obj.write("\n")
                
        parameter["index_size"] = index+1
        parameter["batch_size"] = time

# Release video memory (if it`s available).
torch.cuda.empty_cache()
end = datetime.datetime.now()      
print("Train Time: {}".format(end-start))

# save model
torch.save(net, './result/net_torch_{}.pkl'.format(torch.__version__))
# save parameter.json
with open('./parameter.json', 'w') as file_obj:
    json.dump(parameter, file_obj, sort_keys=True, indent=4, separators=(',', ':'))

