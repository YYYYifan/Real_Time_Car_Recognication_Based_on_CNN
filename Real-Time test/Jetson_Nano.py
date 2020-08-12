# This file is on the Jetson Nano
import torch 
from packages import camera
import time
import datetime
import torchvision.transforms as transforms
import cv2


print("==========Prepare==========")
myCamera = camera.camera(grayscale=True)

net = torch.load("./result/net.pkl")
net = net.cuda()
P_to_T = transforms.Compose([transforms.ToTensor()])


print("==========Start==========")
window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
while cv2.getWindowProperty("CSI Camera", 0) >= 0:
    image = myCamera.take_a_pic()
    cv2.imshow("CSI Camera", image)

    with torch.no_grad():
	    start = time.time()
	    image = myCamera.take_a_pic()
	    # myCamera.save_pic(image)
	    image = P_to_T(image)
	    image = image.cuda()
	    image = image.unsqueeze(0)    

	    outputs = net(image)
	    _, predicted = outputs.max(1)
	    end = time.time()
	    time_cost = end - start    

	    if predicted[0].item() == 0:
		    label = "others"
	    else:
		    label = "BMW"        

	    print("\n{}    OUTPUT: {}, Time Cose: {}\n".format(datetime.datetime.now(), label, round(time_cost, 2)))

    # This also acts as
    keyCode = cv2.waitKey(30) & 0xFF
    # Stop the program on the ESC key
    if keyCode == 27:
        torch.cuda.empty_cache()   
        cv2.destroyAllWindows()
        break

