import torch
import os
import glob
import matplotlib.pyplot as plt
from src.utils.geometry import Geometry, setup
from src.models.fbpnet import FBPNet, load_fbpnet_from_dict
from models.fouriernet import GeneralizedFNO_BP as GFNO_BP
from models.fbps import FBP

# model = load_fbpnet_from_dict("testing.pt", smooth_filters=True)
sd = torch.load("results/gfno_bp0.5-state-dict.pt", map_location="cpu")

model = GFNO_BP.model_from_state_dict(sd)

# model = FBP(g)
# model.load_state_dict(sd)

geometry = model.geometry

print(model)
m2 = model

# train_sinos, train_y, test_sinos, test_y = setup(geometry, num_to_generate=0, train_ratio=0.5, use_realistic=True, data_path="data/kits_phantoms_256.pt")

# m2.visualize_output(test_sinos, test_y, output_location="show")

print("Done")

import pickle
pickle.dump(model, "hello_there.pickle")
# x = 100
