import torch
import os
import glob
import matplotlib.pyplot as plt
from src.utils.geometry import Geometry, setup
from src.models.fbpnet import FBPNet, load_fbpnet_from_dict
from models.fbps import FBP

# model = load_fbpnet_from_dict("testing.pt", smooth_filters=True)
sd = torch.load("testing.pt")
g = Geometry(sd["ar"], sd["phi_size"], sd["t_size"])

model = FBP.model_from_state_dict(sd)
# model = FBP(g)
# model.load_state_dict(sd)


print(model)

train_sinos, train_y, test_sinos, test_y = setup(model.geometry, num_to_generate=0, train_ratio=0.5, use_realistic=True, data_path="data/kits_phantoms_256.pt")

model.visualize_output(test_sinos, test_y, output_location="show")

# x = 100