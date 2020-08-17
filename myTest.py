# -*- coding: utf-8 -*-

import json
from packages import prepare

'''
with open("./parameter.json", 'r') as file_obj:
    parameter = json.load(file_obj)


parameter["TEST"] = "TEST"

with open('./parameter.json', 'w') as file_obj:    
    json.dump(parameter, file_obj, sort_keys=True, indent=4, separators=(',', ':'))
'''
    
    
myImage = prepare.imagePocess(save=False)    