# -*- coding: utf-8 -*-


import PIL
import random
import numpy as np
import json


class imagePocess:
    
    def __init__(self, save: bool=True):
        """
        

        Parameters
        ----------
        save : bool, optional
            The value is control when this class func finish whether save the result.
            Tue default is True

        Returns
        -------
        None.

        """
                
        self.positivePath, self.negetivePath = self.getPath()        
        
        self.get_picture_path()
        self.rotate()
        self.split_images()
        if save:
            self.save()
            
    def getPath(self, front_viwe_path: str= "./data/front_view.npy"):
        """
        

        Parameters
        ----------
        front_viwe_path : TYPE, optional
            The path of .npy file(dictionary type)
            The default is "./data/front_view.npy".

        Returns
        -------
        positivePath : TYPE
            positive path list.
        negetivePath : TYPE
            negetive path list.

        """
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
    
    
    def get_picture_path(self):
        """
        Transfrom label path (.txt) to image path (.png)

        Returns
        -------
        None.

        """
        self.image_paths = self.positivePath + self.negetivePath
        
        for index in range(len(self.image_paths)):
            self.image_paths[index] = self.image_paths[index].replace("label", "image").replace(".txt", ".jpg")
    
    
    def rotate(self):
        """
        Rotate dataset

        Returns
        -------
        None.

        """
        self.images = []
        for image_path in self.image_paths:
            image = PIL.Image.open(image_path).convert("L")
            weight, height = image.size
            
            left_up = [0, 0, int(weight * 0.8), int(height * 0.8)]
            left_down = [0, int(height * 0.2),  int(weight * 0.8), height]
            right_up = [int(weight * 0.2), 0, weight, int(height * 0.8)]
            right_down = [int(weight * 0.2), int(height * 0.2), weight, height]
            center = [int(weight * 0.2), int(height * 0.2), int(weight * 0.8),int(height * 0.8)]
            
            self.images.append(image)
            self.images.append(image.crop(center))
            self.images.append(image.crop(left_up))
            self.images.append(image.crop(left_down))
            self.images.append(image.crop(right_up))
            self.images.append(image.crop(right_down))
            
            
    def split_images(self):
        """
        Split images to train, verification. 

        Returns
        -------
        None.

        """
        len_images = len(self.images)        
        split_point = int(len_images/2)
        print("len_images: {}, split: {}".format(len_images, split_point))
        positive = self.images[:split_point]
        negetive = self.images[split_point:]
        
        
        print("Positive: {}, Negetive: {}".format(len(positive), len(negetive)))
        
        
        split_point = int(len(self.image_paths) / 2 * 0.8)
        split_point = int(split_point * 6)
        
        
        train_positive = positive[:split_point]
        test_positive = positive[split_point:]
        train_negetive = negetive[:split_point]
        test_negetive = negetive[split_point:]
        
        print("P_Train: {}, P_Verification: {}\nT_Train: {}, T_Verification: {}".format(
            len(train_positive), len(train_negetive),
            len(test_positive), len(test_negetive)
            ))
        
        self.train = train_positive + train_negetive
        self.verification = test_positive + test_negetive
        
        # with open("./data/dataset/configure", "w") as file_obj:
            # file_obj.write("{} {}".format(len(train_positive), len(test_positive)))
            
        with open("./parameter.json", 'r') as file_obj:
            parameter = json.load(file_obj)
            
        parameter["len_each_subset_in_train"] = len(train_positive)
        parameter["len_each_subset_in_verification"] = len(test_positive)
    
        with open('./parameter.json', 'w') as file_obj:
            json.dump(parameter, file_obj, sort_keys=True, indent=4, separators=(',', ':'))
            
        
            
    def save(self):
        """
        Save result

        Returns
        -------
        None.

        """
        
        # Train
        buff = []
        for image in self.train:
            buff.append(np.asarray(image))

        np.save("./data/dataset/train.npy", buff)            
        
        # Verification
        buff = []
        for image in self.verification:
            buff.append(np.asarray(image))

        np.save("./data/dataset/verification.npy", buff)     
            
            
            

            
            
            
    
            
                
            