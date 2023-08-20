import torch
import odl
import numpy as np
import odl.contrib.torch as odl_torch
import matplotlib.pyplot as plt
import time
from geometries import FlatFanBeamGeometry, DEVICE

n_phantoms = 3
phantoms = torch.stack(torch.load("data/HTC2022/HTCTestPhantomsFull.pt")).to(DEVICE)
data = torch.stack(torch.load("data/HTC2022/HTCTestDataFull.pt")).to(DEVICE)

print(phantoms.dtype)
print(data.dtype)

reco_space = odl.uniform_discr(
    min_pt=[-40, -40], max_pt=[40, 40], shape=[512, 512],
    dtype='float32')
angle_partition = odl.uniform_partition(0, 2 * np.pi, 720)
detector_partition = odl.uniform_partition(-56, 56, 560)
geometry = odl.tomo.FanBeamGeometry(
    angle_partition, detector_partition, src_radius=410.66, det_radius=143.08)

g = FlatFanBeamGeometry(720, 560, 410.66, 553.74, 112, [-40,40, -40, 40], [512, 512])

# --- Create Filtered Back-projection (FBP) operator --- #


ray_trafo = odl.tomo.RayTransform(reco_space, geometry)
fbp = odl.tomo.fbp_op(ray_trafo, filter_type='Hann', frequency_scaling=0.8)

Ray = odl_torch.OperatorModule(ray_trafo)
print("ray adjoint", ray_trafo.adjoint)
print("bp adjint", ray_trafo.adjoint.adjoint)
bp = odl_torch.OperatorModule(ray_trafo.adjoint)
FBP = odl_torch.OperatorModule(fbp)

start = time.time()
sinos = Ray(phantoms)
print("projection took", time.time()-start,"s")
sinos_g = g.project_forward(phantoms)
recons_odl = 1000*FBP(sinos)

print(g.ram_lak_filter().shape)
print(g.fourier_transform(sinos).shape)

recons_g =  g.fbp_reconstruct(sinos)
filtered_sinos = g.inverse_fourier_transform(g.fourier_transform(sinos)*g.ram_lak_filter()/2)
recons_manual = bp(filtered_sinos)

look_at_ind = 2

mse_fn = lambda diff: torch.mean(diff**2)
print("="*40)

print("MSEs")
print("mse odl", mse_fn(recons_odl-phantoms))
print("mse no det", mse_fn(recons_manual-phantoms))
print("mse gometry", mse_fn(recons_g - phantoms))

plt.subplot(131)
plt.imshow(sinos[look_at_ind].cpu())
plt.title("simulated")
plt.colorbar()
plt.subplot(132)
plt.title("simulated geometry")
plt.imshow(sinos_g[look_at_ind].cpu())
plt.colorbar()
plt.subplot(133)
plt.imshow(data[look_at_ind].cpu())
plt.title("gt")
plt.colorbar()

fig, _ = plt.subplots(2,2)

plt.subplot(221)
plt.imshow(phantoms[look_at_ind].cpu())
plt.colorbar()
plt.title("gt")
plt.subplot(222)
plt.imshow(recons_odl[look_at_ind].cpu())
plt.colorbar()
plt.title("recon")
plt.subplot(223)
plt.title("recon geometry no determinant")
plt.imshow(recons_manual[look_at_ind])
plt.colorbar()
plt.subplot(224)
plt.imshow(recons_g[look_at_ind].cpu())
plt.title("recon g")
plt.colorbar()
plt.show()