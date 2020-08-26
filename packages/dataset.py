# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 06:29:35 2020

@author: duyif
"""
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class Mydataset(Dataset):
    
    def __init__(self, images: dict, split_point: int):       
        """
        

        Parameters
        ----------
        images : dict
            dataset in dictionary type, and inside must be images.
        split_point : int
            where should split images to positive and negetive.

        Returns
        -------
        None.

        """
        self.data = images
        self.transform = transforms.Compose(
            [
                transforms.Resize((512,512)),
                transforms.ToTensor()                
            ]
        )
        
        self.split_point = split_point
        
    def __len__(self):
        """
        Return dataset length
        """
        
        length = 0
        for key, values in self.data.items():
            for value in values:
                length = length + 1
        
                       
        return length


    def __getitem__(self, index):
        """
        

        Parameters
        ----------
        index : index
            index of images.

        Returns
        -------
        image : torch.tensor
            image in tensor type
        label : str
            label of this image

        """
        label = 0 if index > self.split_point - 1 else 1
                
        if label == 1:
            image = self.transform(self.data["Positive"][index])      
        elif label == 0:
            image = self.transform(self.data["Negetive"][index - self.split_point])      
        
        # image = self.transform(self.data[index])        
        # label = 0 if index > self.split_point else 1 # 
    
        return image, label        