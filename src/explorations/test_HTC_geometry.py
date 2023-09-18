import torch 
from utils.data import DTYPE 
from utils.data import DEVICE 
from utils.tools import MSE 
import matplotlib.pyplot as plt 
from geometries import HTC2022_GEOMETRY 
from utils.data import get_htc2022_train_phantoms 

sinos = torch.stack(torch.load("data/HTC2022/HTCTrainingData.pt", map_location=DEVICE)).to(DTYPE) 
sinos=sinos[:,0:720,:] 
phantoms = get_htc2022_train_phantoms() 
HTC_geom = HTC2022_GEOMETRY 
numeric = HTC_geom.project_forward(phantoms) 
mse = MSE(numeric[0],sinos[0]) 
print(mse) 
plt.figure() 
plt.subplot(121) 
plt.imshow(numeric[0].cpu()) 
plt.title("numeric") 
plt.colorbar() 
plt.subplot(122) 
plt.imshow(sinos[0].cpu()) 
plt.title("true") 
plt.colorbar() 
plt.show() 


recon_sinos = HTC_geom.fbp_reconstruct(sinos*35) 
recon_numeric = HTC_geom.fbp_reconstruct(numeric) 
plt.figure() 
plt.subplot(131) 
plt.imshow(recon_numeric[0].cpu()) 
plt.title("numeric") 
plt.colorbar() 
plt.subplot(132) 
plt.imshow(recon_sinos[0].cpu()) 
plt.title("recon sino") 
plt.colorbar()
plt.subplot(133) 
plt.imshow(phantoms[0].cpu()) 
plt.title("phantom") 
plt.colorbar() 
plt.show()