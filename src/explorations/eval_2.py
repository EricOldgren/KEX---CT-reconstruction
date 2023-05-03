import torch
import os
import glob
import matplotlib.pyplot as plt
from src.utils.geometry import Geometry, setup
from src.models.fbpnet import FBPNet, load_fbpnet_from_dict
from models.fbps import FBP

list=[]

for i in range(1000):
    list.append(1+2j)

def test():
    print(torch.abs(torch.tensor(list).to("cuda")))

test()