import torch
import os
import glob
import matplotlib.pyplot as plt
from src.utils.geometry import BasicModel, setup
from src.models.fbpnet import FBPNet, load_fbpnet_from_dict

model = load_fbpnet_from_dict("testing.pt", smooth_filters=True)


train_sinos, train_y, test_sinos, test_y = setup(model.geometry, num_to_generate=20, train_ratio=0.5)

model.visualize_output(test_sinos, test_y, output_location="show")

x = 100