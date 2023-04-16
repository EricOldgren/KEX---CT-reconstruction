import torch
import os
import glob
import matplotlib.pyplot as plt
from src.utils.geometry import BasicModel, setup, Geometry
from src.models.fbpnet import FBPNet, load_fbpnet_from_dict

ar=1

model = load_fbpnet_from_dict(path="results\prev_res ar0.5 4fbp ver3.pt")

geometry = Geometry(ar, 300, 150)

(train_sinos, train_y, test_sinos, test_y) = setup(geometry, num_to_generate=1,train_ratio=0,use_realistic=True,data_path="data/kits_phantoms_256.pt")


model.visualize_output(test_sinos, test_y, output_location="show")

