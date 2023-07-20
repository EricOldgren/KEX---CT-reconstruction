import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import odl
import odl.contrib.torch as odl_torch

import numpy as np
import matplotlib.pyplot as plt

from utils.parallel_geometry import ParallelGeometry, setup, DEVICE, extend_geometry
from utils.moments import SinoMoments
from utils.more_fno import FNO2d

geom = ParallelGeometry(0.5, 300,150)
ext_geom = extend_geometry(geom)
n_moments = 12
smp = SinoMoments(ext_geom, n_moments=n_moments)

mse_fn = lambda diff : torch.mean(diff**2)
n_phantoms = 30
read_data: torch.Tensor = torch.load("data/kits_phantoms_256.pt").moveaxis(0,1).to(DEVICE)
read_data = torch.concat([read_data[1], read_data[0], read_data[2]])
read_data = read_data[:n_phantoms] # -- uncomment to read this data
read_data /= torch.max(torch.max(read_data, dim=-1).values, dim=-1).values[:, None, None]
PHANTOMS = read_data

print("calcing sinos...")
ray_l = odl_torch.OperatorModule(geom.ray)
ext_ray_l = odl_torch.OperatorModule(ext_geom.ray)

SINOS = ray_l(PHANTOMS)
FULL_SINOS = ext_ray_l(PHANTOMS)

dataset = TensorDataset(SINOS, PHANTOMS, FULL_SINOS)
dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

modes_ph = torch.fft.rfftfreq(geom.phi_size, geom.dphi).shape[0]
modes_t = torch.where(geom.fourier_domain < geom.omega)[0].shape[0]
expfno = FNO2d(modes_ph, modes_t, 1, 1, layer_widths=[30,30], verbose=True, dtype=torch.float)

optimizer = torch.optim.Adam(expfno.parameters(), lr=0.01)
N_epochs = 100

print("starting training-...")
for epoch in range(N_epochs):
    batch_mse_sinos, batch_mse_moms = [], []

    for sinos, y, full_sinos in dataloader:
        optimizer.zero_grad()

        filler = expfno(sinos[:, None])[:, 0]
        exp_sinos = torch.concat([sinos, filler], dim=1)

        moms = [smp.get_moment(exp_sinos, ni) for ni in range(n_moments)]
        proj_moms = [smp.project_moment(mom, ni) for ni, mom in enumerate(moms)]
        mse_mom = sum(mse_fn(p_mom-mom) for p_mom, mom in zip(proj_moms, moms)) / n_moments
        
        mse_sinos = mse_fn(exp_sinos - full_sinos)
        loss = mse_sinos + mse_mom*0.01

        loss.backward()
        optimizer.step()
        batch_mse_sinos.append(mse_sinos.item())
        batch_mse_moms.append(mse_mom.item())

    if epoch %  5 == 0:
        print(f"Epoch: {epoch} mse-sino: ", np.mean(batch_mse_sinos), " and mse-mom: ", np.mean(batch_mse_moms))

ind = 10

plt.subplot(121)
plt.imshow(full_sinos[ind])

plt.subplot(122)
plt.imshow(exp_sinos[ind].detach())

plt.show()

print("dine")
