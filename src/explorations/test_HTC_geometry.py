import torch 
from src.geometries.data import DTYPE 
from src.geometries.data import DEVICE 
from utils.tools import MSE 
import matplotlib.pyplot as plt 
from geometries import HTC2022_GEOMETRY 
from src.geometries.data import get_htc2022_train_phantoms 

sinos = torch.stack(torch.load("data/HTC2022/HTCTrainingData.pt", map_location=DEVICE)).to(DTYPE)*35
sinos=sinos[:,0:720,:] 
phantoms = get_htc2022_train_phantoms() 
HTC_geom = HTC2022_GEOMETRY 
numeric = HTC_geom.project_forward(phantoms)
mse = MSE(numeric,sinos) 
print("mse sinos:", mse)

disp_ind = 2
plt.figure() 
plt.subplot(121) 
plt.imshow(numeric[disp_ind].cpu()) 
plt.title("numeric") 
plt.colorbar() 
plt.subplot(122) 
plt.imshow(sinos[disp_ind].cpu()) 
plt.title("true") 
plt.colorbar() 

plt.figure()
plt.imshow((numeric[disp_ind]-sinos[disp_ind]).cpu())
plt.colorbar()
plt.title("diff")

recon_sinos = HTC_geom.fbp_reconstruct(sinos) 
recon_numeric = HTC_geom.fbp_reconstruct(numeric)
print("recon mse:", MSE(recon_sinos, phantoms))
plt.figure() 
plt.subplot(131) 
plt.imshow(recon_numeric[disp_ind].cpu()) 
plt.title("numeric") 
plt.colorbar() 
plt.subplot(132) 
plt.imshow(recon_sinos[disp_ind].cpu()) 
plt.title("recon sino") 
plt.colorbar()
plt.subplot(133) 
plt.imshow(phantoms[disp_ind].cpu()) 
plt.title("phantom") 
plt.colorbar() 
plt.show()