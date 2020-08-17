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

# Determine whether the device uses CPU or GPU for calculation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
# Load convolution neural network model (trained)
net = torch.load('../result/net_torch_{}.pkl'.format(torch.__version__))
# Send model to GPU (if it is available)
net.to(device)


while True:  
    # Get image from scren
    screen = np.array(ImageGrab.grab(bbox=BOX).convert("L"))
    # Use OpenCV2 to show the captured images 
    cv2.imshow("window",screen)
    # Turn off torch.autograd
    with torch.no_grad():            
        start = time.time()
        # Transform PIL images to tensor (and to GPU if available)
        screen = torch.from_numpy(screen).float().to(device)
        # unsqueeze
        screen = screen.unsqueeze(0).unsqueeze(0)
        outputs = net(screen)
        _, predicted = outputs.max(1)
        
        if predicted.item() == 0:
            label = "Others"
        elif predicted.item() == 1:
            label = "BMW"
        
        print("{}\t output: {}, time cost: {}".format(
			datetime.datetime.now(),
			"Others" if predicted.item() == 0 else "BMW",
			round(time.time()-start, 2)
		))
        
    # Display, exit by "ESC"
    keyCode = cv2.waitKey(30) & 0xFF    
    if keyCode == 27:    
        torch.cuda.empty_cache()   
        cv2.destroyAllWindows()
        break