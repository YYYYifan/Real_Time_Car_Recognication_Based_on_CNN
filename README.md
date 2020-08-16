# Real_Time_Car_Recognication_Based_on_CNN

it is used to classification BMW from other vehicles

in PC has 2 way to test camera, 
    
    1. Using PPT + OpenCV2, monitoring desktop.
    2. Based on "1", using DroidCamApp (WebCamera by using phone).

----

## [Accuracy and Loss in training](./Loss_Acc_Visualization.py)

![Acc_and_Loss](./images/Loss_and_Accuracy_in_Training.png)

---

## [In Verificaiton](./verificaiton.py)
Accuracy

torch 1.6: 97.46%, each image needs 0.0006327 sec to classification in PC

torch 1.4: 96.16%

---

## Real-Time Test
### [In PC](./Real-Time-test/PC.py):
![PC_TEST_BMW](./images/PC_TEST_BMW.png)
![PC_TEST_Others](./images/PC_TEST_Others.png)
### [In Jetson_Nano](./Real-Time-test/Jetson_Nano.py):
![indoor_test](./images/indoor_test.jpg)
![indoor_test2](./images/indoor_test2.jpg)