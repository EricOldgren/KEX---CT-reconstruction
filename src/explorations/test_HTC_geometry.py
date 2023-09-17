import torch
from utils.data import DTYPE
from utils.tools import MSE
import matplotlib as plt

from geometries import HTC2022_GEOMETRY

from utils.data import get_htc2022_train_phantoms

sinos = torch.stack(torch.load("data/HTC2022/HTCTestDataFull.pt", map_location="CPU")).to(DTYPE)
phantoms = get_htc2022_train_phantoms()

HTC_geom = HTC2022_GEOMETRY

numeric = HTC_geom.project_forward(phantoms)
mse = MSE(numeric,sinos)
print(mse)

plt.imshow(numeric(0))
plt.imshow(sinos(0))
plt.show()

recon_phantoms = HTC_geom.fbp_reconstruct(sinos)
recon_numeric = HTC_geom.fbp_reconstruct(numeric)