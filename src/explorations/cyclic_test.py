#Try other direction
#Block for FNO with moment constraints
import torch
from torchvision.transforms import Resize
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import odl.contrib.torch as odl_torch
import numpy as np
from utils.geometry import Geometry, setup, BasicModel, extend_geometry
from utils.threaded_training import multi_threaded_training
import random
from models.fbps import FBP, GeneralizedFBP as GFBP
from models.fbpnet import FBPNet
from models.modelbase import ChainedModels
from models.analyticmodels import RamLak, ramlak_filter
from fno_1d import FNO1d
from models.fouriernet import FNO_BP, GeneralizedFNO_BP as GFNO_BP
from models.expnet import ExtrapolatingBP
from utils.moments import SinoMoments


ar = 0.5
n_epochs = 200
LAMBDA = 1e-7 #scale of regularization

geometry = Geometry(ar, 450, 300)
ext_geom = extend_geometry(geometry)

(train_sinos, train_y, test_sinos, test_y) = setup(geometry, num_to_generate=0, train_ratio=0.8, use_realistic=True, data_path="/content/drive/MyDrive/KEX-data/kits_phantoms_256.pt")
#Cyclic
# omgs = 2*np.pi * torch.fft.rfftfreq(2*ext_geom.phi_size, ext_geom.dphi).to(DEVICE)
# modes = omgs.shape[0]
# fno = FNO1d(modes, ext_geom.t_size, ext_geom.t_size, layer_widths=[30,30], dtype=torch.float).to(DEVICE)
gap, t_size = ext_geom.phi_size - geometry.phi_size, geometry.t_size
omgs = 2*np.pi* torch.fft.rfftfreq(geometry.fourier_domain, geometry.dhpi)
fno = FNO1d(omgs.shape[0], t_size, t_size, [30, 30], dtype=torch.float, verbose=True)

ext_modes = torch.where(ext_geom.fourier_domain <= ext_geom.omega)[0].shape[0]
fbp_fno = FNO1d(ext_modes, ext_geom.phi_size, ext_geom.phi_size, layer_widths=[30, 30], verbose=True, dtype=torch.float32)
final_fbp = GFNO_BP(ext_geom, fbp_fno, ext_geom, dtype=torch.float32)
# final_fbp = GFBP(ext_geom) #RamLak(ext_geom)

full_ray_layer = odl_torch.OperatorModule(ext_geom.ray)
full_train_sinos = full_ray_layer(train_y)
full_test_sinos = full_ray_layer(test_y)
N_moments = 5
smp = SinoMoments(ext_geom, n_moments=N_moments)
A = 0.003 #Regularization: smoothing Helgason-Ludwig

dataset = TensorDataset(train_sinos, train_y, full_train_sinos)
dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

optimizer = torch.optim.Adam(list(fno.parameters()) + list(final_fbp.parameters()), lr=0.001)

mse_fn = lambda diff : torch.mean(diff**2)

for epoch in range(n_epochs):

  if epoch % 10 == 0:
    plot_sinos_fno(fno, final_fbp, test_sinos, test_y, full_test_sinos)

  batch_losses, batch_mom_mse, batch_sino_mse = [], [], []
  for sinos, y, full_sinos in dataloader:
    optimizer.zero_grad()

    padded_sinos = torch.nn.functional.pad(sinos, (0,0,0, ext_geom.phi_size-geometry.phi_size), "constant", 0)
    cyclic_sinos = torch.concatenate([padded_sinos, torch.flip(padded_sinos, dims=[-1])], dim=1)

    exp_part =  F.relu(fno(cyclic_sinos.permute(0,2,1)).permute(0,2,1)[:, geometry.phi_size:ext_geom.phi_size])
    exp_sinos = torch.concat([sinos, exp_part], dim=1)
    # exp_sinos[:, :geometry.phi_size] = sinos

    mse_sino = mse_fn(exp_sinos - full_sinos)

    moms = [smp.get_moment(exp_sinos, i) for i in range(N_moments)]
    proj_moms = [smp.project_moment(mom, i) for i, mom in enumerate(moms)]
    mse_mom = sum(mse_fn(mom-p_mom) for mom, p_mom in zip(moms, proj_moms)) / len(moms)

    recon = final_fbp(exp_sinos)
    mse_recon = mse_fn(recon - y)
    batch_losses.append(mse_recon.item()); batch_mom_mse.append(mse_mom.item()); batch_sino_mse.append(mse_sino.item())

    loss = mse_recon + mse_sino + mse_mom * 0.01
    loss.backward()
    optimizer.step()
  
  print(f"epoch {epoch} mse: {sum(batch_losses) / len(batch_losses)} mse-mom: {sum(batch_mom_mse) / len(batch_mom_mse)}, mse-sino: {sum(batch_sino_mse) / len(batch_sino_mse)}")


