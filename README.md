# Real_Time_Car_Recognication_Embedded_System_Based_on_Convolution_Neural_Network

it is used to classification BMW from other vehicles

in PC has 2 way to test this:
    
    1. Using PPT + OpenCV2, monitoring desktop.
    2. Based on "1", using DroidCamApp (WebCamera by using phone).


---

## [Pre Image Process](./packages/prepare.py)

Step1. **Transform** RBG image to grayscale, it can reduce the mount of calculate.

Step2. **Extended** images by select different part of it.

Step3. **Resize** all images to 512 x 512, this is for convorlution neural network (This step is in [dataset.py](./packages/dataset.py))


> After step2, it has 3060 images for each subset ("positive", "negetive")

----

## [Accuracy and Loss in training](./Loss_Acc_Visualization.py)

We use log in training to draw this images.

[Log File](./result/train.log)


![Acc_and_Loss](./images/Loss_and_Accuracy_in_Training.png)

---

## [In Verificaiton](./verificaiton.py)

It has 1536 images in verificaiton dataset, the half of them is BMW car image and leftover is others car images

[Log File](./result/verification.log)

torch 1.5: 96.16%, each image needs 0.0006465 sec to classification in PC

torch 1.4: 95.92%

---

## Real-Time Test
### [In PC](./Real-Time-test/PC.py):
![PC_TEST_BMW](./images/PC_TEST_BMW.png)
![PC_TEST_Others](./images/PC_TEST_Others.png)
### [In Jetson_Nano](./Real-Time-test/Jetson_Nano.py):
![indoor_test](./images/indoor_test.jpg)
![indoor_test2](./images/indoor_test2.jpg)