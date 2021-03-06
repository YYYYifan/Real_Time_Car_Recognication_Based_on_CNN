# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        # self.shortcut = nn.Sequential()
        if stride == 1 and in_planes == out_planes:
            self.shortcut=True
        else:
            self.shortcut=False
            # self.shortcut = nn.Sequential(
            #     nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
            #     nn.BatchNorm2d(out_planes),
            # )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + x if self.shortcut else out
        return out

class MobileNetV2(nn.Module):

    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 2),  
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 128, 3, 2),
           (6, 256, 1, 1)]

    def __init__(self, num_classes=2):
        super(MobileNetV2, self).__init__()        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        self.linear = nn.Linear(512, num_classes)
        self.pool = nn.AdaptiveMaxPool2d(1)
        
        
    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))        
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out