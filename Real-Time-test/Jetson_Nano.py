# This file is on the Jetson Nano
import torch 
from packages import camera
import time
import datetime
import torchvision.transforms as transforms
import cv2


print("==========Prepare==========")
myCamera = camera.camera(grayscale=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = torch.load("./result/net_torch_1.4.0.pkl")
net = net.to(device)
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
	    image = image.to(device)
	    image = image.unsqueeze(0)    

	    outputs = net(image)
	    _, predicted = outputs.max(1)
	    end = time.time()
	    time_cost = end - start    

	    result = "{}\t output: {}, time cost: {}".format(
            datetime.datetime.now(),
            "Others" if predicted.item() == 0 else "BMW",
            round(time.time()-start, 2)
        )
		print(result)

    # Display, exit by "ESC"
    keyCode = cv2.waitKey(30) & 0xFF    
    if keyCode == 27:
        torch.cuda.empty_cache()   
        cv2.destroyAllWindows()
        break

