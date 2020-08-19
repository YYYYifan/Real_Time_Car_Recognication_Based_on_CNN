# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
import json


with open("./parameter.json", 'r') as file_obj:
    parameter = json.load(file_obj)
    index_size = parameter["index_size"]
    batch_size = parameter["batch_size"]
    
with open("./result/train_torch_1.4.0.log") as file_obj:
    buffer = file_obj.readlines()

for index in range(len(buffer)):
    buffer[index] = buffer[index].replace("\n", "")
    buffer[index] = buffer[index].replace(",", "")
    buffer[index] = buffer[index].replace("[", "")
    buffer[index] = buffer[index].replace("]", "")
    buffer[index] = buffer[index].split(" ")
    
result_log = np.asarray(buffer).astype(float)    

loss = []
for index in range(len(result_log)):
    if index % index_size == 0:        
        loss.append(np.sum(result_log[index: index + index_size, 2]) / index_size)

loss = np.asarray(loss)

accuarcy = []
for index in range(len(result_log)):
    if index % index_size == 0:        
        accuarcy.append(np.sum(result_log[index: index + index_size, 3]) / index_size)

accuarcy = np.asarray(accuarcy)


plt.figure(figsize=(6, 4))
acc, = plt.plot(range(1, len(accuarcy)+1), accuarcy)
loss, = plt.plot(range(1, len(loss)+1), loss/batch_size)
plt.legend((acc, loss), ("Accuracy", "Loss"))
plt.title("Loss and Accuaray in Training")
plt.xlabel("Epoch")


def to_percent(temp, position):
    return '%1.0f'%(100*temp) + '%'

plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(to_percent))

plt.savefig("./images/Loss_and_Accuracy_in_Training.png", dpi = 900, pad_inches=0.0)
plt.show()
