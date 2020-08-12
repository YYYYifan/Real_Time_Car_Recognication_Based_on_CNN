# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 06:29:35 2020

@author: duyif
"""
from torch.utils.data import Dataset
import packages.prepare
import torchvision.transforms as transforms


class Mydataset(Dataset):
    
    def __init__(self, images, split_point):        
        self.data = images
        self.transform = transforms.Compose(
            [
                transforms.Resize((512,512)),
                transforms.ToTensor()                
            ]
        )
        
        self.split_point = split_point
        
    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        image = self.transform(self.data[index])        
        label = 0 if index > self.split_point else 1 # 
    
        return image, label        
    
    



if __name__ == "__main__":
    imagePocessed = packages.prepare.imagePocess("../data/front_view.npy")     
    train_dataset = Mydataset(imagePocessed.train_image)
    # tt = test.backgrond()
    # tt.show()
    