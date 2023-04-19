import torch
import odl
import odl.contrib.torch as odl_torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.fouriernet import GeneralizedFNO_BP
from src.models.fbpnet import FBPNet
from src.utils.geometry import Geometry
from src.models.analyticmodels import RamLak
import random as rnd
import math
from src.utils.fno_1d import FNO1d

geometry = Geometry(0.5, 300, 150)
geometry2 = Geometry(0.5, 450, 300)

data: torch.Tensor = torch.load("data\kits_phantoms_256.pt").moveaxis(0,1).to("cuda")
data = torch.concat([data[1], data[0], data[2]])
test_img = data[:600]

def display_result_img(img, model_path_multi=None, model_path_single=None, model_path_fno=None):
    ray_layer = odl_torch.OperatorModule(geometry.ray)
    sinos: torch.Tensor = ray_layer(img)
    ray_layer2 = odl_torch.OperatorModule(geometry2.ray)
    sinos2: torch.Tensor = ray_layer2(img)
    index = rnd.randrange(0,600)
    original_img = test_img[index].cpu()

    modes = torch.where(geometry.fourier_domain <= geometry.omega)[0].shape[0]
    fno = FNO1d(modes, 300, 600, hidden_layer_widths=[40], verbose=True, dtype=torch.float32)
    ext_geom = Geometry(1.0, phi_size=600, t_size=150)
    model_fno = GeneralizedFNO_BP(geometry, fno, ext_geom)
    model_fno.load_state_dict(torch.load(model_path_fno))
    with torch.no_grad():
        recon_fnos = model_fno.forward(sinos)
        recon_fno = recon_fnos[index].to("cpu")

    model_multi = FBPNet(geometry2, 4)
    model_multi.load_state_dict(torch.load(model_path_multi), strict=False)
    with torch.no_grad():
        recon_multis = model_multi.forward(sinos2)
        recon_multi = recon_multis[index].to("cpu")
    
    model_single = FBPNet(geometry2, 1)
    model_single.load_state_dict(torch.load(model_path_single), strict=False)
    with torch.no_grad():
        recon_singles = model_single.forward(sinos2)
        recon_single = recon_singles[index].to("cpu")

    model_analytic = RamLak(geometry2)
    recon_analytic = model_analytic.forward(sinos2)[index].cpu()
    

    plt.subplot(251)
    plt.imshow(original_img, cmap='gray')

    plt.subplot(252)
    plt.imshow(recon_analytic, cmap='gray')
    plt.title("Result using FBP with analytic kernel")
    plt.subplot(257)
    plt.imshow(recon_analytic-original_img, cmap='gray')

    plt.subplot(253)
    plt.imshow(recon_single, cmap='gray')
    plt.title("Result using FBP with a single trained filter")
    plt.subplot(258)
    plt.imshow(recon_single-original_img, cmap='gray')

    plt.subplot(254)
    plt.imshow(recon_multi, cmap='gray')
    plt.title("Result combining FBPs")
    plt.subplot(259)
    plt.imshow(recon_multi-original_img, cmap='gray')

    plt.subplot(255)
    plt.imshow(recon_fno, cmap='gray')
    plt.title("Results using FNO to reconstruct to expand sinogram")
    plt.subplot(2,5,10)
    plt.imshow(recon_fno-original_img,cmap='gray')


    plt.show()

def display_result_sino():
    pass


def test():
    display_result_img(img=test_img, model_path_multi="results\prev_res ar0.5 4fbp 450_300 ver2.pt", model_path_single="results\prev_res ar0.5 1fbp 450_300 ver1.pt", model_path_fno="results\gfno_bp0.5-state-dict.pt")

test()