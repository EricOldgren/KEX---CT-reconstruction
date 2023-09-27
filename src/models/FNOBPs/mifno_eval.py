import torch
import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt

from utils.tools import htc_score, MSE
from src.geometries.data import get_htc2022_train_phantoms
from utils.polynomials import Legendre, Chebyshev
from geometries import HTC2022_GEOMETRY, enforce_moment_constraints
from models.modelbase import load_model_checkpoint, evaluate_batches
from models.FNOBPs.mifno_bp import FNO_SinoExp

checkpoint = load_model_checkpoint("/home/emastr/deep-limited-angle/KEX---CT-reconstruction/data/models/fno_sino_exp_v1.pt", FNO_SinoExp)
model: FNO_SinoExp = checkpoint.model
ar = checkpoint.angle_ratio

geometry = HTC2022_GEOMETRY
PolynomialFamily = Legendre
PHANTOMS = get_htc2022_train_phantoms()
SINOS = geometry.project_forward(PHANTOMS)

la_sinos, known_angles = geometry.zero_cropp_sinos(SINOS, ar, 0)
exp = model.get_extrapolated_sinos(la_sinos, known_angles)
_, known_region = geometry.reflect_fill_sinos(SINOS, known_angles)

# known_angles = (~known_region).sum(dim=-1,dtype=torch.int) == 0
def poc(sinos: torch.Tensor, M: int, K: int, n_enforce: int, n_iters: int): 
    exp = sinos + 0
    for it in range(n_iters):
        coefficients = geometry.series_expand(exp, PolynomialFamily, M, K)
        enforce_moment_constraints(coefficients[:, :n_enforce])
        exp[:, ~known_region] = geometry.synthesise_series(coefficients, PolynomialFamily)[:, ~known_region]

    return exp

MS = [10, 20, 30, 40, 50, 60]
stricts = [5, 10, 20, 25, 30, 35]
disp_ind = 2

fig, initial_mse = evaluate_batches(exp.detach(), SINOS, disp_ind, "sinos")
fig.show()
print("initial mse:", initial_mse)

out = exp + 0

for i in range(len(MS)):
    M = MS[i]
    K = M
    strict = stricts[i]

    out = poc(out, M, K, strict, 10)

    fig, mse = evaluate_batches(out.detach(), SINOS, disp_ind, f"sinos M={M}")

    print(f"MSE after M={M} is:", mse)
    fig.show()


print("AR:", checkpoint.angle_ratio)
print("final MSE:", mse)
fig.show()
plt.show()
