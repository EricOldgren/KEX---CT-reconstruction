import torch
import matplotlib.pyplot as plt
from utils.data import get_htc2022_train_phantoms
from utils.polynomials import Chebyshev, Legendre
from utils.tools import MSE
from geometries import HTC2022_GEOMETRY, CDTYPE, get_moment_mask

geometry = HTC2022_GEOMETRY
ar = 0.25
M, K = 64, 64

phantoms = get_htc2022_train_phantoms()
print("phantoms loaded")
sinos = geometry.project_forward(phantoms)
la_sinos, known_angles = geometry.zero_cropp_sinos(sinos, ar, 0)

N, Nu, Nb = sinos.shape
n_known_u = geometry.n_known_projections(ar)

mask = get_moment_mask(torch.zeros((1,M,K)))
n_coeffs = mask.sum()
print(n_coeffs, M*(M+1)//2)
coefficients = torch.zeros((N, n_coeffs), dtype=CDTYPE, requires_grad=False)

l1 = 1e-3

n_iters = 100

# optimizer = torch.optim.Adam([coefficients], lr=0.3)
# optimizer = torch.optim.SGD([coefficients], lr=1.0)
# optimizer = torch.optim.LBFGS([coefficients])

# def closure():
#     embedding = torch.zeros((N, M, K), dtype=CDTYPE)
#     embedding[:, mask] += coefficients
#     res = geometry.synthesise_series(embedding, Legendre)
#     return MSE(res[:, :n_known_u], la_sinos[:, :n_known_u]) + l1*torch.mean(torch.abs(coefficients)**2)

for it in range(n_iters):

    embedding = torch.zeros((N, M, K), dtype=CDTYPE)
    embedding[:, mask] += coefficients
    res = geometry.synthesise_series(embedding, Legendre)
    loss = MSE(res[:, :n_known_u], la_sinos[:, :n_known_u]) #+ l1*torch.mean(torch.abs(coefficients)**2)

    err = la_sinos - res
    err[:, n_known_u:] *= 0
    coefficients += geometry.series_expand(err, Legendre, M, K)[:, mask]
    print("iter:", it, "loss:", loss.item())

# exp = la_sinos + 0
embedding = torch.zeros((N, M, K), dtype=CDTYPE)
embedding[:, mask] += coefficients
exp = geometry.synthesise_series(embedding, Legendre)

disp_ind = 2
plt.subplot(121)
plt.imshow(sinos[disp_ind])
plt.subplot(122)
plt.title("exp")
plt.imshow(exp[disp_ind].detach())

plt.show()

for i in plt.get_fignums():
    fig = plt.figure(i)
    title = fig._suptitle.get_text() if fig._suptitle is not None else f"fig{i}"
    plt.savefig(f"{title}.png")





