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

sinos = ray_l(phantoms)
full_sinos = ext_ray_l(phantoms)

MOMS_GT = [smp.get_moment(full_sinos, ni) for ni in range(n_moments)]
PROJ_MOMS_GT = [smp.project_moment(mom, ni) for ni, mom in enumerate(MOMS_GT)]
MOMS_MSE_GT  = sum(mse_fn(p_mom-mom) for p_mom, mom in zip(MOMS_GT, PROJ_MOMS_GT))
print("GT mom mse: ", MOMS_MSE_GT.item())

phi_size, t_size, ext_phi_size = geom.phi_size, geom.t_size, ext_geom.phi_size
gap = ext_phi_size - phi_size

pepper = torch.zeros(n_phantoms, gap, t_size, requires_grad=True)
optimizer = torch.optim.Adam([pepper], lr=1.0)
sch = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

for it in range(1000):
    optimizer.zero_grad()

    exp_sinos = torch.concat([sinos, pepper], dim=1)
    mse_sino = mse_fn(full_sinos-exp_sinos)

    moms = [smp.get_moment(exp_sinos, ni) for ni in range(n_moments)]
    proj_moms = [smp.project_moment(mom, ni) for ni, mom in enumerate(moms)]

    loss = sum(mse_fn(p_mom-mom) for p_mom, mom in zip(proj_moms, moms)) / n_moments
    loss.backward()
    optimizer.step()
    if it % 50 == 0:
        print("iteration", it, "mse mom: ", loss.item(), "mse sino: ", mse_sino.item())
        sch.step()

ind = 0
plt.subplot(121)
plt.imshow(full_sinos[ind].cpu())
plt.title("gt")
plt.subplot(122)
plt.imshow(exp_sinos[ind].detach().cpu())
plt.title("exp")

plt.show()

fbp = RamLak(ext_geom)
recons = fbp(exp_sinos)
print(mse_fn(recons-phantoms))

plt.subplot(121)
plt.imshow(phantoms[ind].cpu())

plt.subplot(122)
plt.imshow(recons[ind].detach().cpu())
plt.title("recon")

plt.show()



