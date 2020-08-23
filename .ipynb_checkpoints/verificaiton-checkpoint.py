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
import json


def numpy_to_PIL(images):
    buff = []
    for image in images:
        buff.append(PIL.Image.fromarray(np.uint8(image)))
    return buff        


# Load parameters
with open("./parameter.json", 'r') as file_obj:
    parameter = json.load(file_obj)
    len_each_subset_in_verification = parameter["len_each_subset_in_verification"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load verification dataset.
verification_images = np.load("./data/dataset/verification.npy", allow_pickle=True)
# Transfrom numpy type to PIL type
verification_images = numpy_to_PIL(verification_images)

print("Loading Net")
net = torch.load("./result/net_torch_{}.pkl".format(torch.__version__))
net.to(device)
print("Finish")


print("Preparing images")
verification_dataset = dataset.Mydataset(verification_images, len_each_subset_in_verification)
verification_dataloader = DataLoader(verification_dataset, batch_size=6, shuffle=True)
print("Finish")


len_index = int(verification_dataset.__len__() / 6)
total = 0
correct = 0
start = datetime.datetime.now()

with open("./result/verification.log", "w") as file_obj:
    with torch.no_grad():
        for i,(image,labels) in enumerate(verification_dataloader):
            image = image.to(device)
            labels = labels.to(device)
            outputs = net(image)
            _, predicted = outputs.max(1)
            total += outputs.size(0)
            correct += predicted.eq(labels).sum().item()
            
            log = "idnex: {}/{}".format(i+1, len_index)
            print(log)
            file_obj.write(log + "\n")
    
    torch.cuda.empty_cache()   
    cur_acc = correct / total
    end = datetime.datetime.now()
    total_time_cost = end - start
    each_time_cost = total_time_cost / verification_dataset.__len__()
    
    log = "\nAccuracy:{:.4f}\nTotal time cost: {}\nEach image needs: {}".format(correct / total, total_time_cost, each_time_cost)
    print(log)
    file_obj.write(log)
