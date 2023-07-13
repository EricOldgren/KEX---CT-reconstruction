import torch
import odl.contrib.torch as odl_torch
import os
import glob
import matplotlib.pyplot as plt
from src.utils.geometry import ParallelGeometry, setup, DEVICE
from src.models.fbpnet import FBPNet, load_fbpnet_from_dict
from models.fouriernet import GeneralizedFNO_BP as GFNO_BP
from models.fbps import FBP

# model = load_fbpnet_from_dict("testing.pt", smooth_filters=True)
sd = torch.load("data/gfno-report-test/gfno-ar0.5-450x300.pt", map_location="cpu")

model = GFNO_BP.model_from_state_dict(sd)

# model = FBP(g)
# model.load_state_dict(sd)

geometry = model.geometry

read_data: torch.Tensor = torch.load("data/kits_phantoms_256.pt").moveaxis(0,1).to(DEVICE)
read_data = torch.concat([read_data[1], read_data[0], read_data[2]])
read_data = read_data[500:600] # -- uncomment to read this data
read_data /= torch.max(torch.max(read_data, dim=-1).values, dim=-1).values[:, None, None]

ray_l = odl_torch.OperatorModule(model.geometry.ray)
print("Calculating sinograms...")
sinos = ray_l(read_data)


print(model)

model.visualize_output(sinos, read_data, output_location="show")

print("Done")

# x = 100
