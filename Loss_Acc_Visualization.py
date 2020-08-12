# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 


Epoch_size = 510
batch = 12

file_obj = open("./result/log.txt")
buffer = file_obj.readlines()
file_obj.close()

for index in range(len(buffer)):
    buffer[index] = buffer[index].replace("\n", "")
    buffer[index] = buffer[index].replace(",", "")
    buffer[index] = buffer[index].replace("[", "")
    buffer[index] = buffer[index].replace("]", "")
    buffer[index] = buffer[index].split(" ")
    
result_log = np.asarray(buffer).astype(float)    

loss = []
for index in range(len(result_log)):
    if index % Epoch_size == 0:        
        loss.append(np.sum(result_log[index: index + Epoch_size, 2]) / Epoch_size)

loss = np.asarray(loss)

accuarcy = []
for index in range(len(result_log)):
    if index % Epoch_size == 0:        
        accuarcy.append(np.sum(result_log[index: index + Epoch_size, 3]) / Epoch_size)

accuarcy = np.asarray(accuarcy)


acc, = plt.plot(range(1, len(accuarcy)+1), accuarcy)
loss, = plt.plot(range(1, len(loss)+1), loss/batch)
plt.legend((acc, loss), ("Accuracy", "Loss"))
plt.title("Loss and Accuaray in Training")
plt.xlabel("Epoch")


def to_percent(temp, position):
    return '%1.0f'%(100*temp) + '%'

plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(to_percent))

plt.savefig("./images/Loss_and_Accuracy_in_Training.png", dpi = 600)
plt.show()
