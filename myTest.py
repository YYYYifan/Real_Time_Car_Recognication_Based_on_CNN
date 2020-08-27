from torch.utils.data import DataLoader
import torch
import datetime
import PIL
import numpy as np
import json
from tqdm import tqdm
from packages import dataset


# Load parameters
with open("./parameter.json", 'r') as file_obj:
    parameter = json.load(file_obj)
    len_each_subset_in_verification = parameter["len_each_subset_in_verification"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load verification dataset.
verification_images = np.load("./data/dataset/verification.npy", allow_pickle=True).item()
# Transfrom numpy type to PIL type
# verification_images = numpy_to_PIL(verification_images)


net = torch.load("./result/net_torch_{}.pkl".format(torch.__version__))
net.to(device)




verification_dataset = dataset.Mydataset(verification_images, len_each_subset_in_verification)
verification_dataloader = DataLoader(verification_dataset, batch_size=6, shuffle=True)



len_index = int(verification_dataset.__len__() / 6)
total = 0
correct = 0
start = datetime.datetime.now()

#with open("./result/verification.log", "w") as file_obj:
with torch.no_grad():
    for i,(image,labels) in enumerate(tqdm(verification_dataloader, ncols=128, leave=True)):
        image = image.to(device)
        labels = labels.to(device)
        outputs = net(image)
        _, predicted = outputs.max(1)
        total += outputs.size(0)
        correct += predicted.eq(labels).sum().item()
            
            # log = "idnex: {}/{}".format(i+1, len_index)
            # print(log)
            # file_obj.write(log + "\n")
    
    torch.cuda.empty_cache()   
    cur_acc = correct / total
    end = datetime.datetime.now()
    total_time_cost = end - start
    each_time_cost = total_time_cost / verification_dataset.__len__()
    
    tqdm.write(str(cur_acc))