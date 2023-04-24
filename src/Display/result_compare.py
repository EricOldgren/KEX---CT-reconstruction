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
from statistical_measures import ssim, psnr
from src.models.fbps import FBP

geometry = Geometry(0.25, 450, 300)

data: torch.Tensor = torch.load("data\kits_phantoms_256.pt").moveaxis(0,1).to("cuda")
data = torch.concat([data[1], data[0], data[2]])
test_img = data[-100:]
index = rnd.randrange(0,100)
test_img /= torch.max(torch.max(test_img, dim=-1).values, dim=-1).values[:, None, None]

def display_result_img(img, model_path_multi=None, model_path_single=None, model_path_fno=None):
    ray_layer = odl_torch.OperatorModule(geometry.ray)
    sinos: torch.Tensor = ray_layer(img)
    original_img = test_img[index].cpu()

    #modes = torch.where(geometry.fourier_domain <= geometry.omega)[0].shape[0]
    #fno = FNO1d(modes, 300, 600, hidden_layer_widths=[30 30], verbose=True, dtype=torch.float32)
    #ext_geom = Geometry(1.0, phi_size=1200, t_size=150)
    #model_fno = GeneralizedFNO_BP(geometry, fno, ext_geom)
    #model_fno.load_state_dict(torch.load(model_path_fno),strict=False)
    model_fno = GeneralizedFNO_BP.model_from_state_dict(torch.load(model_path_fno))
    with torch.no_grad():
        recon_fnos = model_fno.forward(sinos)
        recon_fno = recon_fnos[index].to("cpu")
    print("FNO ssim:", ssim(recon_fno,original_img))
    print("FNO psnr:", psnr(recon_fno,original_img))

    model_multi = FBPNet(geometry, 4)
    model_multi.load_state_dict(torch.load(model_path_multi), strict=False)
    with torch.no_grad():
        recon_multis = model_multi.forward(sinos)
        recon_multi = recon_multis[index].to("cpu")
    print("Multi ssim:", ssim(recon_multi,original_img))
    print("Multi psnr:", psnr(recon_multi,original_img))
    
    model_single = FBP(geometry)
    model_single.load_state_dict(torch.load(model_path_single), strict=False)
    with torch.no_grad():
        recon_singles = model_single.forward(sinos)
        recon_single = recon_singles[index].to("cpu")
    print("Single ssim:", ssim(recon_single,original_img))
    print("Single psnr:", psnr(recon_single,original_img))

    
    model_analytic = RamLak(geometry)
    recon_analytic = model_analytic.forward(sinos)[index].cpu()
    
    #fbp = odl.tomo.fbp_op(geometry.ray, padding=False)
    #proj_data = geometry.ray(original_img)
    #recon_analytic = torch.Tensor(fbp(proj_data).asarray())

    print("Analytic ssim:", ssim(recon_analytic,original_img))
    print("Analytic psnr:", psnr(recon_analytic,original_img))
    
    plt.clc()

    plt.subplot(251)
    plt.imshow(original_img, cmap='gray')
    plt.axis('off')

    plt.subplot(252)
    plt.imshow(recon_analytic, cmap='gray')
    plt.axis('off')
    plt.title("Result using FBP with analytic kernel")
    plt.subplot(257)
    plt.imshow(abs(recon_analytic-original_img.numpy()), cmap='gray')
    plt.axis('off')

    plt.subplot(253)
    plt.imshow(recon_single, cmap='gray')
    plt.axis('off')
    plt.title("Result using FBP with a single trained filter")
    plt.subplot(258)
    plt.imshow(abs(recon_single-original_img), cmap='gray')
    plt.axis('off')

    plt.subplot(254)
    plt.imshow(recon_multi, cmap='gray')
    plt.axis('off')
    plt.title("Result combining FBPs")
    plt.subplot(259)
    plt.imshow(abs(recon_multi-original_img), cmap='gray')
    plt.axis('off')

    plt.subplot(255)
    plt.imshow(recon_fno, cmap='gray')
    plt.axis('off')
    plt.title("Results using FNO to reconstruct to expand sinogram")
    plt.subplot(2,5,10)
    plt.imshow(abs(recon_fno-original_img),cmap='gray')
    plt.axis('off')


    plt.show()

def display_result_sino(img, model_path_fno):
    ext_geom = Geometry(1.0, 1800, 300)
    geom = Geometry(0.25, 450, 300)
    ray_layer = odl_torch.OperatorModule(geom.ray)
    sinos: torch.Tensor = ray_layer(img)
    ray_layer_full = odl_torch.OperatorModule(ext_geom.ray)
    sinos_full: torch.Tensor = ray_layer_full(img)

    true_sino = sinos_full[index].detach().cpu()

    # modes = torch.where(geometry.fourier_domain <= geometry.omega)[0].shape[0]
    # fno = FNO1d(modes, 300, 600, hidden_layer_widths=[40], verbose=True, dtype=torch.float32)
    # model_fno = GeneralizedFNO_BP(geometry, fno, ext_geom)
    # model_fno.load_state_dict(torch.load(model_path_fno))
    model_fno = GeneralizedFNO_BP.model_from_state_dict(torch.load(model_path_fno))
    fno_sinos = model_fno.return_sino(sinos)
    fno_sino = fno_sinos[index].detach().cpu()

    print("fno:", torch.min(fno_sino), torch.max(fno_sino))
    print("orginal", torch.min(true_sino), torch.max(true_sino))

    plt.subplot(131)
    plt.imshow(true_sino, cmap='gray')
    plt.title("Full angle sinogram")
    plt.subplot(132)
    plt.imshow(fno_sino, cmap='gray')
    plt.title("FNO reconstructed sinogram")
    plt.subplot(133)
    plt.imshow(abs(true_sino-fno_sino),cmap='gray')
    plt.title("Absolute difference")

    plt.show()


def test():
    display_result_sino(test_img, "results\gfno_bp-ar0.25-state-dict-450x300.pt")
    display_result_img(img=test_img, model_path_multi="results\final ar0.25 multi ver 2.pt", model_path_single="results\final ar0.25 single ver2.pt", model_path_fno="results\gfno_bp-ar0.25-state-dict-450x300.pt")

test()