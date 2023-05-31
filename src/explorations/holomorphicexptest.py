import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import odl
import odl.contrib.torch as odl_torch

from utils.geometry import Geometry, DEVICE, setup, extend_geometry
from utils.moments import SinoMoments
from models.expnet import ExtrapolatingBP
from models.analyticmodels import RamLak

FACTORIALS = [1]
for k in range(1,30):
    FACTORIALS.append(k*FACTORIALS[-1])

NUM_THRESH = 0.001

geom = Geometry(0.5, 450, 300)
ext_geom = extend_geometry(geom)

n_ders = 5
n_parse = 100
dilation = 1
stride = 5

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

phi_size, t_size, ext_phi_size = geom.phi_size, geom.t_size, ext_geom.phi_size
gap = ext_phi_size - phi_size


full_ghat = torch.fft.rfft(full_sinos)

ghat: torch.Tensor = torch.fft.rfft(sinos)
exp_ghat = torch.concat([ghat, torch.zeros(n_phantoms, gap, ghat.shape[-1], device=DEVICE)], dim=1)

curr = phi_size
while curr < ext_phi_size:
    #curr equals the number of angles the extrapolated sinigram is knonw at
    pre_ps = torch.from_numpy(ext_geom.angles[:curr-1][::dilation][-n_parse:]).to(DEVICE)
    pN = ext_geom.angles[curr-1] #last known angle
    A = torch.stack([(pre_ps-pN)**k / FACTORIALS[k] for k in range(1, n_ders+1)]).T.to(DEVICE, dtype=ghat.dtype) + NUM_THRESH
    B = (exp_ghat[:, curr-n_parse-1:curr-1] - exp_ghat[:, None, curr-1]).permute(1,0,2).reshape(n_parse, -1)

    # X: torch.Tensor = torch.linalg.solve(A, B)
    (X, res, rank, sv) = torch.linalg.lstsq(A, B, driver="gelsd")
    X : torch.Tensor = X.reshape(n_ders, n_phantoms, -1).permute(1,0,2) #X[b,d, s] is (d):th derivative / d! of ghat of phantom b at freq s

    nxt = min(curr + stride, ext_phi_size)
    post_ps = torch.from_numpy(ext_geom.angles[curr:nxt]).to(DEVICE, dtype=exp_ghat.dtype)
    C = torch.stack([(post_ps - pN)**k / FACTORIALS[k] for k in range(1, n_ders+1)]).T #C[a,d] is (pa -pN)**d

    exp_ghat[:, curr:nxt, :] = torch.einsum("ad,bds->bas", C, X) + exp_ghat[:, None, curr-1, :]

    curr = nxt

    print("processed: ", curr)


print("MAX GT:", torch.max(torch.abs(full_ghat)))
print("MAX EXP:", torch.max(torch.abs(exp_ghat)))

print("MSE: ", mse_fn(torch.abs(exp_ghat-full_ghat)))

exp_sinos = torch.fft.irfft(exp_ghat)

print("And in spatial space: ")
print("MAX GT:",torch.max(torch.abs(full_sinos)))
print("MAX EXP:", torch.max(torch.abs(exp_sinos)))

print("MSE:", mse_fn(exp_sinos- full_sinos))

ind = 0
plt.subplot(221)
plt.imshow(torch.abs(full_ghat[ind]).cpu(),   vmin=0, vmax=1.0)
plt.title("gt")
plt.subplot(222)
plt.imshow(torch.abs(exp_ghat[ind]).detach().cpu(), vmin=0.0, vmax=1.0)
plt.title("exp")

plt.subplot(223)
plt.imshow(full_sinos[ind].cpu(), vmin=0.0, vmax=1.0)
plt.title("gt")
plt.subplot(224)
plt.imshow(exp_sinos[ind].detach().cpu(), vmin=0.0, vmax=1.0)
plt.title("exp")

plt.show()

# fbp = RamLak(ext_geom)
# recons = fbp(exp_sinos)
# print(mse_fn(recons-phantoms))

# plt.subplot(121)
# plt.imshow(phantoms[ind].cpu())

# plt.subplot(122)
# plt.imshow(recons[ind].detach().cpu())
# plt.title("recon")

# plt.show()