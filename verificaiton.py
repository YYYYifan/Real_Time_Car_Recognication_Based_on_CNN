# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 14:54:59 2020

@author: duyif
"""
from packages import dataset
from torch.utils.data import DataLoader
import torch
import datetime
import PIL
import numpy as np

def numpy_to_PIL(images):
    buff = []
    for image in images:
        buff.append(PIL.Image.fromarray(np.uint8(image)))
    return buff        

with open("./data/dataset/configure") as file_obj:
    points = file_obj.readlines()
    points = points[0].split(" ")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

verification_images = np.load("./data/dataset/verification.npy", allow_pickle=True)
verification_images = numpy_to_PIL(verification_images)

print("Loading Net")
net = torch.load("./result/net_torch_{}.pkl".format(torch.__version__))
net.to(device)
print("Finish")


print("Preparing images")
verification_dataset = dataset.Mydataset(verification_images, int(points[1]))
verification_dataloader = DataLoader(verification_dataset, batch_size=6, shuffle=True)
print("Finish")


total = 0
correct = 0
start = datetime.datetime.now()
with torch.no_grad():
    for i,(image,labels) in enumerate(verification_dataloader):
        image = image.to(device)
        labels = labels.to(device)
        outputs = net(image)
        _, predicted = outputs.max(1)
        total += outputs.size(0)
        correct += predicted.eq(labels).sum().item()
        print("idnex: {} Accuracy: {:.4f}".format(i, correct/total))

torch.cuda.empty_cache()    

end = datetime.datetime.now()    
cur_acc = correct/total
# best_acc = max(cur_acc,best_acc)
print("Accuracy:{:.4f}".format(correct / total))
# print("Best Accuracy:{:.4f}".format( best_acc))