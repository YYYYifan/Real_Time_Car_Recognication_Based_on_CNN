# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 06:29:35 2020

@author: duyif
"""
from torch.utils.data import Dataset
import packages.prepare
import torchvision.transforms as transforms


class Mydataset(Dataset):
    
    def __init__(self, images):        
        self.data = images
        self.transform = transforms.Compose([transforms.ToTensor()])
        
    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        image = self.transform(self.data[index])        
        label = 1 if index > (self.__len__() / 2) else 0 # 
    
        return image, label        
    
    



if __name__ == "__main__":
    imagePocessed = packages.prepare.imagePocess("../data/front_view.npy")     
    train_dataset = Mydataset(imagePocessed.train_image)
    # tt = test.backgrond()
    # tt.show()
    