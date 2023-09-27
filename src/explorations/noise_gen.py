import torch

from utils.tools import DEVICE, DTYPE, CDTYPE, MSE, GIT_ROOT
from src.geometries.data import disc_phantom, get_htc2022_train_phantoms
from geometries import HTC2022_GEOMETRY, htc_sino_var
import matplotlib.pyplot as plt

# phantoms = get_htc2022_train_phantoms()
sinos = torch.stack(torch.load(GIT_ROOT/"data/HTC2022/HTCTrainingData.pt"))[:, :720]
phantoms = HTC2022_GEOMETRY.fbp_reconstruct(sinos)
# sinos = HTC2022_GEOMETRY.project_forward(phantoms)

noisy = sinos + torch.randn(sinos.shape, device=DEVICE, dtype=DTYPE)*htc_sino_var

o = noisy
noisy = HTC2022_GEOMETRY.rotate_sinos(noisy, 97)
noisy = HTC2022_GEOMETRY.rotate_sinos(noisy, -97)

recons = HTC2022_GEOMETRY.fbp_reconstruct(noisy)

print(MSE(o, noisy))
print(MSE(noisy, sinos))
print(MSE(phantoms, recons))

disp_ind = 2
plt.subplot(121)
plt.imshow(noisy[disp_ind].cpu())
plt.colorbar()
plt.title("noisy")
plt.subplot(122)
plt.imshow(sinos[disp_ind].cpu())
plt.colorbar()
plt.title("gt")

plt.figure()
plt.subplot(121)
plt.imshow(recons[disp_ind].cpu())
plt.title("recon")
plt.subplot(122)
plt.imshow(phantoms[disp_ind].cpu())
plt.title("gt")

plt.show()



