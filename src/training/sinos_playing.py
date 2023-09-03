import torch

import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt

from utils.polynomials import Legendre, Chebyshev
from utils.tools import MSE
from utils.data import get_htc2022_train_phantoms
from geometries import HTC2022_GEOMETRY as geometry, enforce_moment_constraints, get_moment_mask
from geometries.geometry_base import naive_sino_filling
from models.modelbase import evaluate_batches

PHANTOMS = get_htc2022_train_phantoms()
SINOS = geometry.project_forward(PHANTOMS)
PolynomialFamily = Chebyshev
M, K = 100, 50


ar = 0.25
sinos_la, known = geometry.zero_cropp_sinos(SINOS, ar, 0)
sinos_la, known = geometry.reflect_fill_sinos(sinos_la, known)
known_betas = (~known).sum(dim=-1) == 0
filled = naive_sino_filling(sinos_la, known_betas)
exp_naive = sinos_la + 0
exp_naive[:, ~known] = filled[:, ~known]
disp_ind = 0
exp = sinos_la + 0

coefficients = geometry.series_expand(exp_naive, PolynomialFamily, M, K)
strict_coeffs = enforce_moment_constraints(coefficients+0)

exp_strict = sinos_la+0
exp_strict = geometry.synthesise_series(strict_coeffs, PolynomialFamily)
exp_full = sinos_la+0
exp_full[:, ~known] = geometry.synthesise_series(coefficients, PolynomialFamily)[:, ~known]

coefficients_clean = geometry.series_expand(sinos_la, PolynomialFamily, M, K)
enforce_moment_constraints(coefficients_clean)
exp_clean_strict = sinos_la+0
exp_clean_strict[:, ~known] = geometry.synthesise_series(coefficients_clean, PolynomialFamily)[:, ~known]

disp_ind = 1
clean_strict_fig, clean_strict_mse = evaluate_batches(exp_clean_strict, SINOS, disp_ind, "clean strict")
strict_fig, strict_mse = evaluate_batches(exp_strict, SINOS, disp_ind, "strict")
full_fig, full_mse = evaluate_batches(exp_full, SINOS, disp_ind, "full")
naive_fig, naive_mse = evaluate_batches(exp_naive, SINOS, disp_ind, "naive")

print("naive mse:", MSE(sinos_la, SINOS))
print("diff naive - strict:", MSE(exp_strict, sinos_la))
print("clean strict mse:", clean_strict_mse)
print("strict mse:", strict_mse)
print("full mse:", full_mse)

plt.show()




