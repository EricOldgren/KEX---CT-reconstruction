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
from utils.data_generator import unstructured_random_phantom


FACTORIALS = [1]
for k in range(1,30):
    FACTORIALS.append(k*FACTORIALS[-1])

NUM_THRESH = 0.001


geom = ParallelGeometry(0.5,450, 300)
ext_geom = extend_geometry(geom)

# phantoms = torch.from_numpy(np.array(unstructured_random_phantom(geom.reco_space)))[None]
phantoms = torch.from_numpy(odl.phantom.transmission.shepp_logan(ext_geom.reco_space, True).asarray())[None]
n_ders = 5
n_parse = 100
dilation = 1
stride = 5

# mse_fn = lambda diff : torch.mean(diff**2)
# n_phantoms = 10
# read_data: torch.Tensor = torch.load("data/kits_phantoms_256.pt").moveaxis(0,1).to(DEVICE)
# read_data = torch.concat([read_data[1], read_data[0], read_data[2]])
# read_data = read_data[:n_phantoms] # -- uncomment to read this data
# read_data /= torch.max(torch.max(read_data, dim=-1).values, dim=-1).values[:, None, None]
# phantoms = read_data

ray_l = odl_torch.OperatorModule(geom.ray)
ext_ray_l = odl_torch.OperatorModule(ext_geom.ray)

sinos = ray_l(phantoms)
full_sinos = ext_ray_l(phantoms)
full_sinos = torch.concat([full_sinos, torch.flip(full_sinos, (-1,)) ], dim=1)

phi_size, t_size, ext_phi_size = geom.phi_size, geom.t_size, ext_geom.phi_size
gap = ext_phi_size - phi_size

# exp_sinos = torch.concat([sinos, torch.zeros(n_phantoms, gap, t_size, device=DEVICE, dtype=sinos.dtype)], dim=1)
exp_sinos = F.pad(sinos, (0,0,0,gap))
exp_sinos = torch.concat([exp_sinos, torch.flip(exp_sinos, (-1,))], dim=1)

plt.subplot(121)
plt.imshow(exp_sinos[0].cpu())

plt.subplot(122)
plt.imshow(full_sinos[0].cpu())

plt.show()

Ks = torch.fft.fftfreq(ext_geom.phi_size, ext_geom.dphi)
sigs = ext_geom.fourier_domain
k = Ks 

fourg = torch.fft.fft2(F.pad(exp_sinos, (0,0,0,0)))
full_fourg = torch.fft.fft2(F.pad(full_sinos, (0,0,0,0)))

#print("MSE:", torch.mean(torch.abs(full_fourg-fourg)**2))

plt.subplot(121)
plt.imshow(torch.abs(fourg[-1].cpu()), vmin=0.0, vmax=1.0)
plt.title("exp")

plt.subplot(122)
plt.imshow(torch.abs(full_fourg[-1].cpu()), vmin=0.0, vmax=1.0)
sigs = np.linspace(-500,500, 2000)
plt.plot(sigs + full_fourg.shape[2] / 2, full_fourg.shape[1] / 2 + sigs*ext_geom.rho, "r", label="k=rho*sig")
plt.plot(sigs + full_fourg.shape[2] / 2, full_fourg.shape[1] / 2 - sigs*ext_geom.rho, "r")
plt.title("GT")

plt.show()