import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Union
from statistics import mean
from math import log10
from tqdm import tqdm
print(0, "Memory allocated:", torch.cuda.memory_allocated()>>30)
import matplotlib.pyplot as plt

from utils.polynomials import Chebyshev
from utils.tools import MSE
from geometries.data import get_synthetic_htc_phantoms, HTC2022_GEOMETRY
from geometries.extrapolation import PrioredSinoFilling, RidgeSinoFiller

from sklearn.model_selection import train_test_split

geometry = HTC2022_GEOMETRY
PHANTOMS = get_synthetic_htc_phantoms()
SINOS = geometry.project_forward(PHANTOMS)

PHANTOMS, VALIDATION_PHANTOMS, SINOS, VALIDATION_SINOS = train_test_split(PHANTOMS, SINOS, test_size=0.5)
p_eval = 0.1
print(1, " data loaded and split. Memory allocated:", torch.cuda.memory_allocated()>>30)
ar = 0.25
M, K = 50, 50
r = 10
l2_reg = 0.01
la_sinos, known_angles = geometry.zero_cropp_sinos(SINOS, ar, 0)
la_validation_sinos, _ = geometry.zero_cropp_sinos(VALIDATION_SINOS, ar, 0)
print(2, "Sino cropping done. Memory allocated:", torch.cuda.memory_allocated()>>30)

filler = PrioredSinoFilling(geometry, known_angles, M, K, Chebyshev).fit_prior(SINOS)
filler_no_mu = PrioredSinoFilling(geometry, known_angles, M, K, Chebyshev).fit_prior(SINOS, use_mu=False)
filler_ridge = RidgeSinoFiller(geometry, known_angles, M, K, Chebyshev)
print(3, "Sinofiller objects initialized and priors calculated. Memory allocated:", torch.cuda.memory_allocated()>>30)

def eval_filler(filler: Union[PrioredSinoFilling, RidgeSinoFiller], la_sinos: torch.Tensor, sinos: torch.Tensor, phantoms: torch.Tensor, l2_reg: float, r: int):
    dataset = TensorDataset(la_sinos, sinos, phantoms)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    mse_sinos, mse_phantoms = [], []
    for la, full, ph in dataloader:
        if isinstance(filler, PrioredSinoFilling):
            exp = filler.forward(la, r, l2_reg)
        else:
            exp = filler.forward(la, l2_reg)
        recons = geometry.fbp_reconstruct(exp)
        mse_sinos.append(MSE(exp, full).item())
        mse_phantoms.append(MSE(recons, ph).item())

    return mean(mse_sinos), mean(mse_phantoms)

rs = [1, 10, 100, 500, None]
log_l2_reg_range = torch.linspace(-5, 5, 30)

for ri in rs:
    sino_mses, recon_mses = [], []
    sino_mses_no_mu, recon_mses_no_mu = [], []
    for log_l2_reg in tqdm(log_l2_reg_range, desc="evaluating filler"):
        l2_reg = 10**log_l2_reg
        _, la_eval, _, full_eval, _, phantoms_eval = train_test_split(la_validation_sinos, VALIDATION_SINOS, VALIDATION_PHANTOMS, test_size=p_eval)
        mse_sinos, mse_phantoms = eval_filler(filler, la_eval, full_eval, phantoms_eval, l2_reg, ri)
        sino_mses.append(log10(mse_sinos))
        recon_mses.append(log10(mse_phantoms))
        mse_sinos, mse_phantoms = eval_filler(filler_no_mu, la_eval, full_eval, phantoms_eval, l2_reg, ri)
        sino_mses_no_mu.append(log10(mse_sinos))
        recon_mses_no_mu.append(log10(mse_phantoms))

    plt.figure(0)
    plt.subplot(121)
    plt.plot(log_l2_reg_range, sino_mses, label=f"{ri if ri is not None else 'all'}")
    plt.subplot(122)
    plt.plot(log_l2_reg_range, sino_mses_no_mu, label=f"{ri if ri is not None else 'all'}")
    plt.figure(1)
    plt.subplot(121)
    plt.plot(log_l2_reg_range, recon_mses, label=f"{ri if ri is not None else 'all'}")
    plt.subplot(122)
    plt.plot(log_l2_reg_range, recon_mses_no_mu, label=f"{ri if ri is not None else 'all'}")

sino_mses_ridge, recon_mses_ridge = [], []
for log_l2_reg in tqdm(log_l2_reg_range, desc="evaluating filler"):
    l2_reg = 10**log_l2_reg
    _, la_eval, _, full_eval, _, phantoms_eval = train_test_split(la_validation_sinos, VALIDATION_SINOS, VALIDATION_PHANTOMS, test_size=p_eval)
    mse_sinos, mse_phantoms = eval_filler(filler_ridge, la_eval, full_eval, phantoms_eval, l2_reg, ri)
    sino_mses_ridge.append(log10(mse_sinos))
    recon_mses_ridge.append(log10(mse_phantoms))

plt.figure(0)
plt.subplot(122)
plt.plot(log_l2_reg_range, sino_mses_ridge, label=f"ridge")
plt.figure(1)
plt.subplot(122)
plt.plot(log_l2_reg_range, recon_mses_ridge, label=f"ridge")


titles = ["sino", "recons"]
for i in plt.get_fignums():
    plt.figure(i)
    plt.subplot(121)
    plt.legend()
    plt.subplot(122)
    plt.legend()
    plt.savefig(titles[i])