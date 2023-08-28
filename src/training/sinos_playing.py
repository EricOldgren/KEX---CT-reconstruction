import torch

import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt

from utils.polynomials import Legendre, Chebyshev
from utils.tools import get_htc2022_train_phantoms, MSE
from geometries import HTC2022_GEOMETRY as geometry
from geometries.geometry_base import naive_sino_filling
from models.modelbase import evaluate_batches

PHANTOMS = get_htc2022_train_phantoms()
SINOS = geometry.project_forward(PHANTOMS)
        


ar = 0.25
sinos_la, known = geometry.zero_cropp_sinos(SINOS, ar, 0)
sinos_la, known = geometry.reflect_fill_sinos(sinos_la, known)
known_betas = (~known).sum(dim=-1) == 0
filled = naive_sino_filling(sinos_la, known_betas)
sinos_la[:, ~known] = filled[:, ~known]
disp_ind = 0
exp = sinos_la + 0
# recons_full = geometry.fbp_reconstruct(SINOS)
# fig0, mse0 = evaluate_batches(recons_full, PHANTOMS, disp_ind, "full recon")
# fig0.show()

for it in range(20):

    recons = geometry.fbp_reconstruct(exp)
    if it % 5 == 0:
        sino_figit, mseit = evaluate_batches(exp, SINOS, disp_ind, f"exp it: {it}")
        figit, mseit_recon = evaluate_batches(recons, PHANTOMS, disp_ind, f"reconstruction it: {it}")
        print("iteration:", it+1, "sino mse:", mseit, "recon mse:", mseit_recon)
    exp[:, ~known] = geometry.project_sinos(exp, Legendre, 30)[:, ~known]



plt.show()




