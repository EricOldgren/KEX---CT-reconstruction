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
import cv2

geometry = Geometry(0.5, 450, 300)

data: torch.Tensor = torch.load("data\kits_phantoms_256.pt").moveaxis(0,1).to("cuda")
data = torch.concat([data[1], data[0], data[2]])
test_img = data[500:600]
index = 50#rnd.randrange(0,100)
test_img /= torch.max(torch.max(test_img, dim=-1).values, dim=-1).values[:, None, None]

def display_result_img(img, model_path_multi=None, model_path_single=None, model_path_fno=None):
    ray_layer = odl_torch.OperatorModule(geometry.ray)
    sinos: torch.Tensor = ray_layer(img)
    original_img = test_img[index].cpu()

    model_fno = GeneralizedFNO_BP.model_from_state_dict(torch.load(model_path_fno))
    with torch.no_grad():
        recon_fnos = model_fno.forward(sinos)
        recon_fno = recon_fnos[index].to("cpu")


    #model_multi = FBPNet(geometry, 4,use_padding=True)
    #model_multi.load_state_dict(torch.load(model_path_multi), strict=False)
    model_multi = FBPNet.model_from_state_dict(torch.load(model_path_multi))
    with torch.no_grad():
        recon_multis = model_multi.forward(sinos)
        recon_multi = recon_multis[index].to("cpu")

    #model_single = FBP(geometry,use_padding=True)
    #model_single.load_state_dict(torch.load(model_path_single), strict=False)
    model_single=FBP.model_from_state_dict(torch.load(model_path_single))
    with torch.no_grad():
        recon_singles = model_single.forward(sinos)
        recon_single = recon_singles[index].to("cpu")


    
    model_analytic = RamLak(geometry)
    recon_analytic = model_analytic.forward(sinos)[index].cpu()
    
    #fbp = odl.tomo.fbp_op(geometry.ray, padding=False)
    #proj_data = geometry.ray(original_img)
    #recon_analytic = torch.Tensor(fbp(proj_data).asarray())



    plt.subplot(251)
    plt.title("Original image")
    plt.imshow(original_img, cmap='gray')
    plt.axis('off')

    plt.subplot(252)
    plt.imshow(recon_analytic, cmap='gray')
    plt.axis('off')
    plt.imsave("res_img\Analytic0.5.png",recon_analytic,cmap="gray",vmin=0,vmax=1)
    plt.title("Result using FBP with analytic kernel")
    plt.subplot(257)
    plt.imshow(abs(recon_analytic-original_img.numpy()), cmap='gray')
    plt.axis('off')
    plt.imsave("res_img\Analytic0.5diff.png",abs(recon_analytic-original_img.numpy()), cmap='gray',vmin=0,vmax=1)

    plt.subplot(253)
    plt.imshow(recon_single, cmap='gray')
    plt.axis('off')
    plt.imsave("res_img\Single0.5.png",recon_single,cmap="gray",vmin=0,vmax=1)
    plt.title("Result using FBP with a single trained filter")
    plt.subplot(258)
    plt.imshow(abs(recon_single-original_img), cmap='gray')
    plt.axis('off')
    plt.imsave("res_img\Single0.5diff.png",abs(recon_single-original_img.numpy()), cmap='gray',vmin=0,vmax=1)

    plt.subplot(254)
    plt.imshow(recon_multi, cmap='gray')
    plt.axis('off')
    plt.imsave("res_img\Multi0.5.png",recon_multi,cmap="gray",vmin=0,vmax=1)
    plt.title("Result combining 4 FBPs")
    plt.subplot(259)
    plt.imshow(abs(recon_multi-original_img), cmap='gray')
    plt.axis('off')
    plt.imsave("res_img\Multi0.5diff.png",abs(recon_multi-original_img.numpy()), cmap='gray',vmin=0,vmax=1)

    plt.subplot(255)
    plt.imshow(recon_fno, cmap='gray')
    plt.axis('off')
    plt.imsave("res_img\Fno0.5.png",recon_fno,cmap="gray",vmin=0,vmax=1)
    plt.title("Results using FNO")
    plt.subplot(2,5,10)
    plt.imshow(abs(recon_fno-original_img),cmap='gray')
    plt.axis('off')
    plt.imsave("res_img\Fno0.5diff.png",abs(recon_fno-original_img.numpy()), cmap='gray',vmin=0,vmax=1)


    plt.show()

def display_single(img, model_path):
    ray_layer = odl_torch.OperatorModule(geometry.ray)
    sinos: torch.Tensor = ray_layer(img)
    original_img = test_img[index].cpu()

    model = FBPNet.model_from_state_dict(torch.load(model_path))
    with torch.no_grad():
        recon_imgs = model.forward(sinos)
        recon_img = recon_imgs[index].to("cpu")
    print("FNO ssim:", ssim(recon_img,original_img))
    print("FNO psnr:", psnr(recon_img,original_img))

    plt.imshow(recon_img,cmap="gray")
    plt.title("Result using")
    plt.axis('off')

    plt.show()

    plt.imshow(abs(recon_img-original_img),cmap="gray")
    plt.title("Difference")
    plt.axis('off')

    plt.show()


def display_result_sino(img, model_path_fno):
    ext_geom = Geometry(1.0, 1800, 300)
    geom = geometry
    ray_layer = odl_torch.OperatorModule(geom.ray)
    sinos: torch.Tensor = ray_layer(img)
    ray_layer_full = odl_torch.OperatorModule(ext_geom.ray)
    sinos_full: torch.Tensor = ray_layer_full(img)

    true_sino = sinos_full[index].detach().cpu()

    model_fno = GeneralizedFNO_BP.model_from_state_dict(torch.load(model_path_fno))
    fno_sinos = model_fno.return_sino(sinos)
    fno_sino = fno_sinos[index].detach().cpu()

    print("fno:", torch.min(fno_sino), torch.max(fno_sino))
    print("orginal", torch.min(true_sino), torch.max(true_sino))

    plt.subplot(131)
    plt.imshow(true_sino, cmap='gray')
    plt.imsave("res_img\Filtered_sino_org0.25.png",true_sino,cmap="gray")
    plt.title("Full angle sinogram")
    plt.subplot(132)
    plt.imshow(fno_sino, cmap='gray')
    plt.imsave("res_img\Fno_sino0.25.png",fno_sino,cmap="gray",vmin=-7,vmax=7)
    plt.title("FNO reconstructed filtered sinogram")
    plt.subplot(133)
    plt.imshow(abs(true_sino-fno_sino),cmap='gray')
    plt.imsave("res_img\sino_diff0.25.png",abs(true_sino-fno_sino),cmap="gray",vmin=-7,vmax=7)
    plt.title("Absolute difference")

    plt.show()


def sino_images():
    #display_result_sino(test_img, "results\gfno_bp-ar0.25-state-dict-450x300.pt")
    #display_result_img(test_img, model_path_multi="results\Final-ar0.5-multi-ver-3.pt", model_path_single="results\Final-ar0.5-single-ver-3.pt", model_path_fno="results\gfno_bp-ar0.5-state-dict-450x300.pt")
    
    #display_single(test_img,"results\Final-ar0.25-multi-ver-3.pt")

    geom = Geometry(1.0,450,300)
    ray_layer = odl_torch.OperatorModule(geom.ray)
    sinos: torch.Tensor = ray_layer(test_img)
    model_fno = GeneralizedFNO_BP.model_from_state_dict(torch.load("results\gfno_bp-ar1.0-state-dict-450x300.pt"))
    with torch.no_grad():
        #fno_sino = model_fno.fno(sinos)[index].cpu().numpy()
        fno_sino = model_fno.return_sino(sinos)[index].cpu().numpy()

    analytic = RamLak(geom)
    sinos = analytic.return_filtered_sino(sinos)
    sino = sinos[index].cpu().numpy()

    #sino = np.pad(sino,[(0,150),(0,0)],"constant")
    sino = cv2.resize(sino,dsize=(300,450))
    fno_sino = cv2.resize(fno_sino,dsize=(300,450))


    plt.imshow(sino,cmap="gray")
    plt.axis("off")
    plt.imsave("SinogramsFinalReport\Sino1.0.png",sino,cmap="gray")
    plt.show()
    plt.imshow(fno_sino,cmap="gray",vmin=-5,vmax=5)
    plt.axis("off")
    plt.imsave("SinogramsFinalReport\Fno_sino1.0.png",fno_sino,cmap="gray",vmin=-5,vmax=5)
    plt.show()
    plt.imshow(abs(sino-fno_sino),cmap="gray",vmin=-5,vmax=5)
    plt.axis("off")
    plt.imsave("SinogramsFinalReport\Sino_diff1.0.png",abs(sino-fno_sino),cmap="gray",vmin=-5,vmax=5)
    plt.show()