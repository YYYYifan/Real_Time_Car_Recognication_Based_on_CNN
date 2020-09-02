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

batch_size = 12

# Load parameters
with open("./parameter.json", 'r') as file_obj:
    parameter = json.load(file_obj)
    len_each_subset_in_verification = parameter["len_each_subset_in_verification"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load verification dataset.
verification_images = np.load("./data/dataset/verification.npy", allow_pickle=True).item()
# Transfrom numpy type to PIL type
# verification_images = numpy_to_PIL(verification_images)

print("Loading Net")
net = torch.load("./result/net_torch_{}.pkl".format(torch.__version__))
net.to(device)
print("Finish")


print("Preparing images")
verification_dataset = dataset.Mydataset(verification_images, len_each_subset_in_verification)
verification_dataloader = DataLoader(verification_dataset, batch_size=batch_size, shuffle=True)
print("Finish")


len_index = int(verification_dataset.__len__() / batch_size)
total = 0
correct = 0
start = datetime.datetime.now()


confusion_martix = np.zeros([2,2], dtype = int)
with open("./result/verification_torch_{}.log".format(torch.__version__), "w") as file_obj:
    with torch.no_grad():
        for i,(image,labels) in enumerate(verification_dataloader):
            image = image.to(device)
            labels = labels.to(device)
            outputs = net(image)
            predicted = outputs.max(1).indices
        
            log = "index: {}/{}".format(i+1, len_index)
            print(log)
            
            for index in range(batch_size):
                if (labels[index] == predicted[index]).item():
                    if predicted[index] == 1: confusion_martix[1, 1] = confusion_martix[1, 1] + 1 # TN
                    if predicted[index] == 0: confusion_martix[0, 0] = confusion_martix[0, 0] + 1 # TP
                else:
                    if predicted[index] == 1: confusion_martix[1, 0] = confusion_martix[1, 0] + 1 # FP
                    if predicted[index] == 0: confusion_martix[0, 1] = confusion_martix[0, 1] + 1 # FN
            
            log = "index: {}/{}\n".format(i+1, len_index)
            print(log)
            file_obj.write(log)
            
            
    
    torch.cuda.empty_cache()   
    
    end = datetime.datetime.now()
    total_time_cost = end - start
    each_time_cost = total_time_cost / verification_dataset.__len__()
    
    n_sample = np.sum(confusion_martix)
    
    #if feature_type == "SIFT" and distance_type == "Cosine" and k == 1:
    #    print(confusion_martix)
    Accuracy = (confusion_martix[1,1] + confusion_martix[0,0]) / n_sample
    Precision = confusion_martix[1,1] / (confusion_martix[1,1] + confusion_martix[0,1])
    if confusion_martix[1,1] == 0 or confusion_martix[1,0] == 0:
        Recall = 0
    else:
        Recall = confusion_martix[1,1] / (confusion_martix[1,1] + confusion_martix[1,0])
    if Recall == 0:
        F1_Score = 0
    else:                
        F1_Score = 2 * Precision * Recall /  (Precision + Recall)
    # print([Accuracy, Precision, Recall, F1_Score])        
    
    log = "\nAccuracy: {:.4f}\nPrecision: {:.4f}\nRecall: {:.4f}\nF1_Score: {:.4f}\nTotal time cost: {}\nEach image needs: {}".format(Accuracy, Precision, Recall, F1_Score, total_time_cost, each_time_cost)
    print(log)
    file_obj.write(log)
    file_obj.write("\n" + str([Accuracy, Precision, Recall, F1_Score]))  
    file_obj.write("\n" + str(confusion_martix))
