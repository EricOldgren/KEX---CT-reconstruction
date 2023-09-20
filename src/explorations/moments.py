import torch
import matplotlib.pyplot as plt
print("phantoms loaded")

from utils.data import get_htc2022_train_phantoms
print("phantoms loaded")
from utils.polynomials import Chebyshev, Legendre
print("phantoms loaded")
from utils.tools import MSE
print("phantoms loaded")
from geometries import HTC2022_GEOMETRY
print("phantoms loaded")

geometry = HTC2022_GEOMETRY
ar = 0.25
M, K = 32, 32


phantoms = get_htc2022_train_phantoms()
print("phantoms loaded")
sinos = geometry.project_forward(phantoms)
la_sinos, known_angles = geometry.zero_cropp_sinos(sinos, ar, 0)

N, Nu, Nb = sinos.shape
n_known_u = geometry.n_known_projections(ar)

coefficients = torch.zeros((N, M, K))

n_iters = 100
optimizer = torch.optim.Adam([coefficients], lr=0.01)
# optims = torch.optim.LBFGS([coefficients])

for it in range(n_iters):
    optimizer.zero_grad()

    res = geometry.synthesise_series(coefficients, Legendre)
    loss = MSE(res[:, :n_known_u], la_sinos[:, :n_known_u])

    loss.backward()
    optimizer.step()
    print("iter:", it, "loss:", loss.item())

exp = la_sinos + 0
exp[:, n_known_u:] = geometry.synthesise_series(coefficients, Legendre)[:, n_known_u:]

disp_ind = 2
plt.subplot(121)
plt.imshow(sinos[disp_ind])
plt.subplot(122)
plt.title("exp")
plt.imshow(exp[disp_ind])



for i in plt.get_fignums():
    fig = plt.figure(i)
    title = fig._suptitle.get_text() if fig._suptitle is not None else f"fig{i}"
    plt.savefig(f"{title}.png")



