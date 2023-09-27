import torch
from torch.utils.data import TensorDataset, DataLoader


import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt

from utils.tools import htc_score
from src.geometries.data import get_htc2022_train_phantoms, get_kits_train_phantoms
from utils.polynomials import Legendre, Chebyshev
from geometries import HTC2022_GEOMETRY, DEVICE, DTYPE, CDTYPE, FlatFanBeamGeometry, enforce_moment_constraints, get_moment_mask

geometry = HTC2022_GEOMETRY
# geometry = FlatFanBeamGeometry(720, 560, 410.66, 543.74, 112, [-40,40, -40, 40], [256, 256])
PHANTOMS = get_htc2022_train_phantoms()
# PHANTOMS = get_kits_train_phantoms()[:50]
SINOS = geometry.project_forward(PHANTOMS)
N = PHANTOMS.shape[0]
PolFamily = Legendre #Chebyshev

n_deg = 120
n_trig_degs = 60

# projected = geometry.synthesise_series(geometry.series_expand(SINOS, PolFamily, n_deg, n_trig_degs), PolFamily)
projected = geometry.moment_project(SINOS, PolFamily, n_deg)
print("projected loss:", torch.mean((projected-SINOS)**2))
coefficients = geometry.series_expand(SINOS, PolFamily, n_deg, n_trig_degs)
orig_coefficients = coefficients + 0
strict_coeffs = coefficients + 0
enforce_moment_constraints(strict_coeffs)
hlcc_mask = get_moment_mask(coefficients)
coefficients[:, ~hlcc_mask] *= 0
to_train = coefficients[:, hlcc_mask] + 0
to_train.requires_grad_(True)

dataset = TensorDataset(PHANTOMS, SINOS)
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

optimizer = torch.optim.Adam([to_train], lr=0.09)

n_iters = 200
for it in range(n_iters):
    for phantom_batch, sino_batch in dataloader:
        optimizer.zero_grad()

        coefficients_it = torch.zeros((N, n_deg, n_trig_degs), device=DEVICE, dtype=CDTYPE)
        coefficients_it[:, hlcc_mask] += to_train
        exp = geometry.synthesise_series(coefficients_it, PolFamily)
        recons = geometry.fbp_reconstruct(exp)

        loss = torch.mean((recons-PHANTOMS)**2)
        loss.backward()
        optimizer.step()

    print("iteratio:", it, "loss:",loss.item())


disp_ind = 2

enforced_synthesised = geometry.synthesise_series(strict_coeffs, PolFamily)
print("MSE of invalid coefficients:", torch.mean(torch.abs(strict_coeffs-orig_coefficients)**2))
print("projected loss (sino domain):", torch.mean((projected-SINOS)**2))
print("projected via enforce loss (sino domain):", torch.mean((enforced_synthesised-SINOS)**2))

plt.figure()
plt.suptitle("extrapolated")
plt.imshow(exp[disp_ind].cpu().detach())
plt.colorbar()
plt.figure()
plt.suptitle("coefficients")
plt.imshow(coefficients[disp_ind].cpu().detach().abs())
plt.colorbar()
plt.figure()
plt.suptitle("gt")
plt.imshow(SINOS[disp_ind].cpu())
plt.colorbar()
plt.figure()
plt.suptitle("projected")
plt.imshow(projected[disp_ind].cpu())
plt.colorbar()

recons_proj = geometry.fbp_reconstruct(projected)
full_recons = geometry.fbp_reconstruct(SINOS)
enforced_recon = geometry.fbp_reconstruct(enforced_synthesised)
print("full recon loss:",torch.mean((full_recons-PHANTOMS)**2))
print("recon projected orig implementation MSE:", torch.mean((recons_proj-PHANTOMS)**2))
print("recon projected via enforce MSE:", torch.mean((enforced_recon-PHANTOMS)**2))
print("recon enforced scores:", htc_score(enforced_recon > 0.5, PHANTOMS.to(torch.bool)))
print("EXP scores:", htc_score(recons > 0.5, PHANTOMS.to(torch.bool)))

plt.figure()
plt.suptitle("gt")
plt.imshow(PHANTOMS[disp_ind].cpu())
plt.colorbar()
plt.figure()
plt.suptitle("exp recon")
plt.imshow(recons[disp_ind].cpu().detach())
plt.colorbar()
plt.figure()
plt.suptitle("exp recon threshholded")
plt.imshow(recons[disp_ind].cpu().detach() > 0.5)
plt.colorbar()
plt.figure()
plt.suptitle("enforced recon threshholded")
plt.imshow(enforced_recon[disp_ind].cpu() > 0.5)
plt.colorbar()
plt.figure()
plt.suptitle("projected recon")
plt.imshow(recons_proj[disp_ind].cpu())
plt.colorbar()
plt.figure()
plt.suptitle("full recon")
plt.imshow(full_recons[disp_ind].cpu())
plt.colorbar()


plt.show()



