import torch
import matplotlib.pyplot as plt
from utils.data import get_htc2022_train_phantoms
from utils.polynomials import Chebyshev, Legendre
from utils.tools import MSE
from geometries import HTC2022_GEOMETRY, CDTYPE, get_moment_mask, DEVICE
from tqdm import tqdm

geometry = HTC2022_GEOMETRY
ar = 0.4
M, K = 64, 64

phantoms = get_htc2022_train_phantoms()
print("phantoms loaded")
sinos = geometry.project_forward(phantoms)
la_sinos, known_angles = geometry.zero_cropp_sinos(sinos, ar, 0)

mask = get_moment_mask(torch.zeros((1,M,K)))
n_coeffs = mask.sum()
mks = torch.cartesian_prod(torch.arange(0,M), torch.arange(0,K)).to(DEVICE).reshape(M, K, 2)[mask]
N, Nu, Nb = sinos.shape
n_known_u = geometry.n_known_projections(ar)

B = torch.zeros((n_coeffs, n_coeffs), device=DEVICE, dtype=CDTYPE)

print("constructing matrix")
for i, (m, k) in tqdm(enumerate(mks)):
    inp = torch.zeros((1,M,K)).to(DEVICE, dtype=CDTYPE)
    inp[:, m, k] = 1
    x = geometry.synthesise_series(inp, Legendre)
    x[:, n_known_u:] *= 0
    B[:, i] = geometry.series_expand(x, Legendre, M, K)[0, mask]
print("Matrix constructed")

cs = torch.linalg.solve(B, geometry.series_expand(la_sinos, Legendre, M, K)[:, mask].reshape(N, n_coeffs))
embedding = torch.zeros((N, M, K), device=DEVICE, dtype=CDTYPE)
embedding[:, mask] += cs

exp = geometry.synthesise_series(embedding)




# exp = la_sinos + 0ding, Legendre)
print("exp error:", MSE(exp, sinos))

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





