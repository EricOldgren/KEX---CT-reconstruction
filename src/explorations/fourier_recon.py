import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import odl
import odl.contrib.torch as odl_torch

from utils.parallel_geometry import ParallelGeometry, DEVICE, setup, extend_geometry
from utils.moments import SinoMoments
from models.expnet import ExtrapolatingBP
from models.analyticmodels import RamLak


NUM_THRESH = 0.001

geom = ParallelGeometry(0.5, 500, 300)
ext_geom = extend_geometry(geom)


mse_fn = lambda diff : torch.mean(diff**2)
n_phantoms = 10
read_data: torch.Tensor = torch.load("data/kits_phantoms_256.pt").moveaxis(0,1).to(DEVICE)
read_data = torch.concat([read_data[1], read_data[0], read_data[2]])
read_data = read_data[:n_phantoms] # -- uncomment to read this data
read_data /= torch.max(torch.max(read_data, dim=-1).values, dim=-1).values[:, None, None]
phantoms = read_data

f = torch.fft.fft2(F.pad(phantoms[0], (100,100,100,100)))

plt.imshow(phantoms[0])

plt.show()

plt.imshow(torch.abs(f), vmin=0.0, vmax=1.0)

plt.show()