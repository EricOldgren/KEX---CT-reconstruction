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

geom = ParallelGeometry(0.5, 450, 300)
ext_geom = extend_geometry(geom)
n_moments = 12
smp = SinoMoments(ext_geom, n_moments=n_moments)

mse_fn = lambda diff : torch.mean(diff**2)
n_phantoms = 2
read_data: torch.Tensor = torch.load("data/kits_phantoms_256.pt").moveaxis(0,1).to(DEVICE)
read_data = torch.concat([read_data[1], read_data[0], read_data[2]])
read_data = read_data[:n_phantoms] # -- uncomment to read this data
read_data /= torch.max(torch.max(read_data, dim=-1).values, dim=-1).values[:, None, None]
phantoms = read_data

ray_l = odl_torch.OperatorModule(geom.ray)
ext_ray_l = odl_torch.OperatorModule(ext_geom.ray)

SINOS_GT = ray_l(phantoms)


pepper_recon = torch.zeros(n_phantoms, *geom.reco_space.shape, requires_grad=True)
optimizer = torch.optim.Adam([pepper_recon], lr=0.1)
# sch = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

for it in range(1000):
    optimizer.zero_grad()

    sinos_pepper = ray_l(pepper_recon)

    loss = mse_fn(SINOS_GT-sinos_pepper)
    loss.backward()
    optimizer.step()

    if it % 50 == 0:
        print("iteration", it, "mse partial sinos: ", loss.item(), "mse recon: ", mse_fn(phantoms-pepper_recon).item() )
        # sch.step()

ind = 0
plt.subplot(121)
plt.imshow(phantoms[ind].cpu())
plt.title("gt")
plt.subplot(122)
plt.imshow(pepper_recon[ind].detach().cpu())
plt.title("exp")

plt.show()