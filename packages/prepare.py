# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 06:59:57 2020

@author: duyif
"""

import numpy as np
import random
import PIL


class imagePocess:
    
    def __init__(self, front_viwe_path = "./data/front_view.npy"):        
        self.positivePath, self.negetivePath = self.getPath(front_viwe_path)        
        self.backgrond_size, self.image_location =\
            self.get_backgrond_size(self.positivePath, self.negetivePath)
    
        self.positiveImages, self.negetiveImages = self.resize_image()
        random.shuffle(self.positiveImages)
        random.shuffle(self.negetiveImages)
        self.spilit_images()
        
                
    
    def getPath(self, front_viwe_path = "./data/front_view.npy"):
        front_viwe = np.load(front_viwe_path, allow_pickle=True).item()
        
        positivePath = front_viwe["81"]  # NOTE: "81" is BMW
        
        negetivePath = []
        for key, paths in front_viwe.items():
            if key != "81":       
                for path in paths:
                    negetivePath.append(path)
        
        random.shuffle(negetivePath)
        
        negetivePath = negetivePath[:len(positivePath)]
        
        return positivePath, negetivePath
    
    
    def get_backgrond_size(self, positivePath, negetivePath):
        image_location = []
        
        for path in positivePath + negetivePath:
            file_obj = open(path)
            buffer = file_obj.readlines()
            file_obj.close()
            
            image_location.append(buffer[2].replace("/n", "").split(" "))
        
        image_location = np.asarray(image_location).astype(int)    
        backgrond_size = np.asarray(
            [
              np.max(image_location[:, 0]),  # X1
              np.max(image_location[:, 1]),  # y2
              np.max(image_location[:, 2]),  # X2
              np.max(image_location[:, 3]),  # Y2   
                ])  
        
        return backgrond_size, image_location
        
  
    def create_backgrond(self, backgrond_size = tuple((1024, 768))):
        return PIL.Image.new("L", tuple(backgrond_size[2:4]), color = 255)                   
    
   
    def resize_image(self):
        paths = self.positivePath + self.negetivePath
        for index in range(len(paths)):
            paths[index] = paths[index].replace("label", "image").replace(".txt", ".jpg")
            
        split_point = int(len(paths) / 2)
        
        for index in range(len(paths)):
            image = PIL.Image.open(paths[index]).convert("L")
            background = self.create_backgrond(self.backgrond_size)
            image = image.crop(self.image_location[index, 0:4])
            background.paste(image, (int((self.backgrond_size[2] - image.size[0])/2),   # Weight
                                     int((self.backgrond_size[3] - image.size[1])/2)))  # Height
            paths[index] = background

        return paths[0: split_point], paths[split_point: len(paths)]            
        
    
    def spilit_images(self):
        # NOTE: 95% of images for train, 5% images for verification
        len_images = len(self.positiveImages)
        split_point = int(len_images - (len_images * 0.05))
        
        
        self.train_image = []
        for image in self.positiveImages[0 : split_point] + self.negetiveImages[0 : split_point]:
            self.train_image.append(image)  
        '''
        self.train_image = {
            "BMW": self.positiveImages[0 : split_point],
            "Others": self.negetiveImages[0 : split_point]
            }        
        '''                  
       
        self.verification_image = []
        for image in self.positiveImages[split_point: len_images] + self.negetiveImages[split_point: len_images]:
            self.verification_image.append(image)
        '''
        self.verification_image = {
            "BMW": self.positiveImages[split_point: len_images],
            "Others": self.negetiveImages[split_point: len_images]
            }
        '''
        
      

if __name__ == "__main__":
    test = imagePocess(front_viwe_path = "../data/front_view.npy")