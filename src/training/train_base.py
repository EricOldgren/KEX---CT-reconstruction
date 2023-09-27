import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("WebAgg")

from utils.tools import MSE, GIT_ROOT
from src.geometries.data import get_htc2022_train_phantoms, get_kits_train_phantoms, get_htclike_train_phantoms, generate_htclike_batch
from utils.polynomials import Legendre, Chebyshev
from geometries import FlatFanBeamGeometry, DEVICE, HTC2022_GEOMETRY, ParallelGeometry
from models.modelbase import save_model_checkpoint, plot_model_progress
from models.fbps import AdaptiveFBP as AFBP
from models.SerieBPs.series_bp1 import Series_BP
from statistics import mean


PHANTOM_DATA = torch.concat([get_htc2022_train_phantoms(), generate_htclike_batch(5,5)])
geometry = HTC2022_GEOMETRY
ar = 0.25
M, K = 100, 50

SINO_DATA = geometry.project_forward(PHANTOM_DATA)



model = Series_BP(geometry, ar, M, K, Legendre.key)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, betas=(0.8, 0.9))

dataset = TensorDataset(SINO_DATA, PHANTOM_DATA)
dataloader = DataLoader(dataset, batch_size=12, shuffle=True)

mse_fn = lambda diff : torch.mean(diff**2)
n_epochs = 1500
for epoch in range(n_epochs):
    batch_losses, batch_sino_losses, batch_recon_losses = [], [], []
    for sino_batch, phantom_batch in dataloader:
        optimizer.zero_grad()

        start_ind = 0 #torch.randint(0, geometry.n_projections, (1,)).item()
        la_sinos, known_angles = geometry.zero_cropp_sinos(sino_batch, ar, start_ind)
        exp_sinos = model.get_extrapolated_sinos(la_sinos, known_angles)

        loss_sino_domain = MSE(exp_sinos, sino_batch)
        # recons = geometry.fbp_reconstruct(exp_sinos)
        # loss_recon_domain = MSE(recons, phantom_batch)
        loss = loss_sino_domain

        loss.backward()
        optimizer.step()

        batch_losses.append(loss.cpu().item())
        batch_sino_losses.append(loss_sino_domain.cpu().item())
        batch_recon_losses.append(-1)
    
    print("Epoch:", epoch+1, "loss is:", mean(batch_losses), "sino loss is:", mean(batch_sino_losses), "recon loss is:", mean(batch_recon_losses), "Memory:", torch.cuda.memory_allocated(DEVICE))

save_model_checkpoint(model, optimizer, loss, ar, f"series_bp_v1.pt")
print("checkpoint saved")
disp_ind = 2
plot_model_progress(model, SINO_DATA, known_angles, None, PHANTOM_DATA, disp_ind, "Series BP", True)

