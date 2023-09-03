print("apa")
import torch
import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt
print("apa")

from utils.tools import htc_score, MSE
print("apa")
from utils.data import get_htc2022_train_phantoms
from utils.polynomials import Legendre, Chebyshev
from geometries import HTC2022_GEOMETRY, enforce_moment_constraints
from models.modelbase import load_model_checkpoint, evaluate_batches
from models.FNOBPs.mifno_bp import FNO_SinoExp
print("apa")

checkpoint = load_model_checkpoint("/home/emastr/deep-limited-angle/KEX---CT-reconstruction/fno_sino_exp_v1.pt", FNO_SinoExp)
model: FNO_SinoExp = checkpoint.model
ar = checkpoint.angle_ratio

geometry = HTC2022_GEOMETRY
PolynomialFamily = Legendre
M, K = 100, 50
PHANTOMS = get_htc2022_train_phantoms()
SINOS = geometry.project_forward(PHANTOMS)


la_sinos, known_angles = geometry.zero_cropp_sinos(SINOS, ar, 0)
exp = model.get_extrapolated_sinos(la_sinos, known_angles)

n_iters = 40
_, known_region = geometry.reflect_fill_sinos(SINOS, known_angles)
# known_angles = (~known_region).sum(dim=-1,dtype=torch.int) == 0
for it in range(n_iters):
    print("iteration:", it, "MSE:", MSE(exp, SINOS))
    coefficients = geometry.series_expand(exp, PolynomialFamily, M, K)
    enforce_moment_constraints(coefficients[:, :10]) #enforce the ten first moment constraints
    exp[:, ~known_region] = geometry.synthesise_series(coefficients, PolynomialFamily)[:, ~known_region]

disp_ind = 2
fig, mse = evaluate_batches(exp.detach(), SINOS, disp_ind, "sinos")

print("final MSE:", mse)
fig.show()
plt.show()
