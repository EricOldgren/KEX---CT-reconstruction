import torch
import os
import glob
import matplotlib.pyplot as plt
from src.utils.geometry import BasicModel, setup
from src.models.fbpnet import FBPNet, load_fbpnet_from_dict

model = load_fbpnet_from_dict("test-path.pt")


print(model)

x = 100