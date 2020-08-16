from packages import prepare
from packages import dataset
from packages import models

from torch.utils.data import DataLoader
import torch
import datetime

train_times = 40
learnRate = 0.001
batch_size = 12

myImage = prepare.imagePocess(save=True)

with open("./data/dataset/configure") as file_obj:
    points = file_obj.readlines()
    points = points[0].split(" ")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = dataset.Mydataset(myImage.train, int(points[0]))
# verification_dataset = dataset.Mydataset(imagePocessed.verification_image)
# Create Dataloder
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# verification_dataloader = DataLoader(verification_dataset, batch_size=2, shuffle=True)


net = models.MobileNetV2(2)    
net.to(device)


optimizer = torch.optim.SGD(net.parameters(), lr=learnRate, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


print("Start Train")
start = datetime.datetime.now()
logFile = open('./result/log.txt', 'w')
for time in range(train_times):
    adjust_learning_rate(optimizer,time,learnRate)
    total = 0
    correct = 0
    net.train(True)
    for i,(image,labels) in enumerate(train_dataloader):
            image = image.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # print("input shape",image.shape)
            outputs = net(image)
            # print(outputs.shape,labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print(time, i, data_select, loss)
            _, predicted = outputs.max(1)
            total += outputs.size(0)
            # print(targets)
            # print(predicted)
            correct += predicted.eq(labels).sum().item()
            Accuracy = correct/total
            print("Epoch: {}/{}, idx:{} Loss:{:.4f} Accuracy:{:.4f}".format(time+1,train_times,i,loss,Accuracy))
            logData = [time, i, loss.item(), Accuracy]
            logFile.write(str(logData))
            logFile.write("\n")

torch.cuda.empty_cache()
end = datetime.datetime.now()      
print("Train Time: {}".format(end-start))

logFile.close()
torch.save(net, './result/net_torch_{}.pkl'.format(torch.__version__))