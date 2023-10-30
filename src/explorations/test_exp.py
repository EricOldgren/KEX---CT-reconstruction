import torch
import matplotlib.pyplot as plt

from utils.polynomials import Chebyshev
from utils.tools import MSE
from geometries.data import get_synthetic_htc_phantoms, HTC2022_GEOMETRY
from geometries.extrapolation import PrioredSinoFilling

from sklearn.model_selection import train_test_split

geometry = HTC2022_GEOMETRY
PHANTOMS = get_synthetic_htc_phantoms()
SINOS = geometry.project_forward(PHANTOMS)

PHANTOMS, VALIDATION_PHANTOMS, SINOS, VALIDATION_SINOS = train_test_split(PHANTOMS, SINOS)
ar = 0.25
M, K = 50, 50
r = 10
l2_reg = 0.01
la_sinos, known_angles = geometry.zero_cropp_sinos(SINOS, ar, 0)
la_validation_sinos, _ = geometry.zero_cropp_sinos(VALIDATION_SINOS, ar, 0)

filler = PrioredSinoFilling(geometry, known_angles, M, K, Chebyshev).fit_prior(SINOS)

exp = filler.forward(la_sinos, r, l2_reg)
recons = geometry.fbp_reconstruct(exp)
validation_exp = filler.forward(la_validation_sinos, r, l2_reg)
validation_recons = geometry.fbp_reconstruct(validation_exp)

print("train sino MSE:", MSE(SINOS, exp))
print("train phantom MSE:", MSE(PHANTOMS, recons))
print("sino MSE:", MSE(VALIDATION_SINOS, validation_exp))
print(MSE("phantom MSE:", validation_recons, VALIDATION_PHANTOMS))

plt.subplot(121)
plt.imshow(VALIDATION_PHANTOMS[0].cpu())
plt.subplot(122)
plt.imshow(validation_recons[0].cpu())

plt.savefig("testing_exp.png")

