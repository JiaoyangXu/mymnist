#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jiaoyangxu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self):
        self.layer1 = nn.linear (784, 200)
        self.layer2 = nn.linear (200,10)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = f.relu(self.layer2(x))
        out = softmax(x, dim=0)
        return out
    
