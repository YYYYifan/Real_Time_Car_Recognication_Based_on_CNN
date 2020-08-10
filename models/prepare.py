# -*- coding: utf-8 -*-

import numpy as np
import random
import PIL


class imageProcess:
    def __init__(self, front_viwe_path:str="./data/front_view.npy"):        
        self.car_border = (800, 600)
        print("__init__")

        self.positivePath, self.negetivePath = self.getPath(front_viwe_path)      
        self.backgrond_size, self.image_location =\
            self.get_backgrond_size(self.positivePath, self.negetivePath)

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

    def get_image_size(self, positivePath, negetivePath):
        paths = self.positivePath + self.negetivePath
        for index in range(len(paths)):
            paths[index] = paths[index].replace("label", "image").replace(".txt", ".jpg")
        
        image_size = []
        for index in range(len(paths)):
            image = PIL.Image.open(paths[index])
            image_s = [image.size[0],image.size[1]]
            image_size.append(image_s)

    
        return image_size       
        