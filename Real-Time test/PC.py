# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 16:52:10 2020

@author: duyif
"""
import cv2
import torch
import datetime
import time
import numpy as np
from PIL import ImageGrab

BOX=(0,25,880,640)

net = torch.load("../result/net.pkl")
net.cuda()

while True:    
    screen=np.array(ImageGrab.grab(bbox=BOX).convert("L"))
    cv2.imshow("window",screen)
    with torch.no_grad():    
        start = time.time()
        screen = torch.from_numpy(screen).float().cuda()
        screen = screen.unsqueeze(0).unsqueeze(0)
        outputs = net(screen)
        _, predicted = outputs.max(1)
        
        if predicted.item() == 0:
            label = "Others"
        elif predicted.item() == 1:
            label = "BMW"
        
        print("{}\t output: {}, time cost: {}".format(datetime.datetime.now(), label, round(time.time()-start, 2)))
        
    # This also acts as
    keyCode = cv2.waitKey(30) & 0xFF
    # Stop the program on the ESC key
    if keyCode == 27:
        torch.cuda.empty_cache()   
        cv2.destroyAllWindows()
        break