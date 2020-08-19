# -*- coding: utf-8 -*-

import json
from packages import prepare
import PIL
import numpy as np
'''
with open("./parameter.json", 'r') as file_obj:
    parameter = json.load(file_obj)


parameter["TEST"] = "TEST"

with open('./parameter.json', 'w') as file_obj:    
    json.dump(parameter, file_obj, sort_keys=True, indent=4, separators=(',', ':'))
'''
    
    
# myImage = prepare.imagePocess(save=False)    
'''
from packages import prepare
help(prepare.imagePocess)
'''
'''

obj_folder = "C:\\Users\\duyif\\OneDrive\\Birmingham\\GraduationProject\\Figure\\Image Processing\\extend\\"

images = []
image = PIL.Image.open(r'D:\CompCars\data\image\81\92\2014\8d42b4ffe292f6.jpg')
image.save("{}{}.png".format(obj_folder, "1.original"))
image = image.convert("L")
weight, height = image.size

left_up = [0, 0, int(weight * 0.8), int(height * 0.8)]
left_down = [0, int(height * 0.2),  int(weight * 0.8), height]
right_up = [int(weight * 0.2), 0, weight, int(height * 0.8)]
right_down = [int(weight * 0.2), int(height * 0.2), weight, height]
center = [int(weight * 0.2), int(height * 0.2), int(weight * 0.8),int(height * 0.8)]

images.append(image)
images.append(image.crop(center))
images.append(image.crop(left_up))
images.append(image.crop(left_down))
images.append(image.crop(right_up))
images.append(image.crop(right_down))

images[0].save("{}{}.png".format(obj_folder, "2.grayscale"))
images[1].save("{}{}.png".format(obj_folder, "3.center"))
images[2].save("{}{}.png".format(obj_folder, "4.left_up"))
images[3].save("{}{}.png".format(obj_folder, "5.left_down"))
images[4].save("{}{}.png".format(obj_folder, "6.right_up"))
images[5].save("{}{}.png".format(obj_folder, "7.right_down"))
'''

def man_dis(p1, p2):
    return np.sum(np.abs(p1 - p2))

dataset = np.zeros([10, 157], dtype = float)
sample = np.zeros([1, 157], dtype = float)
sample[0, 0] = 15

for i, data in enumerate(dataset):
    data[0] = i + 1


arr_dis = []


for data in dataset:
    arr_dis.append(man_dis(sample, data))

list.sort(arr_dis)
    
