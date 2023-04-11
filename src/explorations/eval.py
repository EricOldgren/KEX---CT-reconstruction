import torch
import os
import glob
import matplotlib.pyplot as plt
from src.utils.geometry import Geometry, setup
from src.models.fbpnet import FBPNet, load_fbpnet_from_dict
from models.fbps import FBP

# model = load_fbpnet_from_dict("testing.pt", smooth_filters=True)
sd = torch.load("data/fbpnet/fbpnet.pt")
sd["n_fbps"] = 2
for prefix in ("fbp0", "fbp1"):
    sd[f"{prefix}.ar"] = 0.5; sd[f"{prefix}.phi_size"] = sd["phi_size"]; sd[f"{prefix}.t_size"] = sd["t_size"]
g = Geometry(sd["ar"], sd["phi_size"], sd["t_size"])

model = FBPNet.model_from_state_dict(sd)
# model = FBP(g)
# model.load_state_dict(sd)


print(model)
g2 = Geometry(1.0, sd["phi_size"], sd["t_size"])
m2 = model.convert(g2)

train_sinos, train_y, test_sinos, test_y = setup(g2, num_to_generate=0, train_ratio=0.5, use_realistic=True, data_path="data/kits_phantoms_256.pt")

m2.visualize_output(test_sinos, test_y, output_location="show")

print("Done")

# x = 100