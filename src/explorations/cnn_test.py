import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import odl.contrib.torch as odl_torch

from utils.geometry import Geometry, DEVICE, setup
from utils.moments import SinoMoments
from models.expnet import CNNExtrapolatingBP



AR = 0.25
N_epochs = 50

geom = Geometry(AR, phi_size=50, t_size=30)
model = CNNExtrapolatingBP(geom)
N_moments = 5
smp = SinoMoments(model.extended_geometry, n_moments=5)

train_sinos, train_y, test_sinos, test_y = setup(geom, num_to_generate=0, use_realistic=True, data_path="data/kits_phantoms_256.pt")
Ray_layer_FULL = odl_torch.OperatorModule(model.extended_geometry.ray)
print("calculating full sinos...")
train_sinos_full, test_sinos_full = Ray_layer_FULL(train_y), Ray_layer_FULL(test_y)

dataset = TensorDataset(train_sinos, train_y, train_sinos_full)
dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
mse_fn = lambda diff: torch.mean(diff**2)

print("Starting trainig loop...")
for epoch in range(N_epochs):
    batch_mse_recon, batch_mse_sino, batch_mse_mom = [], [], []
    if epoch % 20 == 10:
        model.visualize_output(test_sinos, test_y, test_sinos_full, output_location="show")
    for sinos, y, full_sinos in dataloader:
        optimizer.zero_grad()
        exp_sinos = model.extrapolate(sinos)
        moms = [smp.get_moment(exp_sinos, ni) for ni in range(N_moments)]
        proj_moms = [smp.project_moment(mom, ni) for ni, mom in enumerate(moms)]
        mse_mom = sum([mse_fn(mom - p_mom) for mom, p_mom in zip(moms, proj_moms)]) / N_moments

        mse_sino = mse_fn(exp_sinos-full_sinos)
        batch_mse_sino.append(mse_sino.item()); batch_mse_mom.append(mse_mom.item())

        loss = mse_sino + mse_mom * 0.01
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} mse_sino: {np.mean(batch_mse_sino)} mse mom: {np.mean(batch_mse_mom)}")

        