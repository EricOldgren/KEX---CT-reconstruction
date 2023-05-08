import torch
import odl
import odl.contrib.torch as odl_torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.fouriernet import GeneralizedFNO_BP
from src.models.expnet import FNOExtrapolatingBP
from src.models.fbpnet import FBPNet
from src.utils.geometry import Geometry, DEVICE
from src.models.analyticmodels import RamLak
import random as rnd
import math
from src.utils.fno_1d import FNO1d
from skimage.metrics import structural_similarity as ssim_import
from src.models.fbps import FBP



def MSE(img, true_img):
    return torch.mean((img-true_img)*(img-true_img))

def psnr(img,true_img):
    return 20*math.log10(torch.max(true_img))-10*math.log10(MSE(img,true_img))

def ssim(image, true_image):
    image=image.detach().cpu().numpy()
    true_image=true_image.detach().cpu().numpy()
    return ssim_import(image, true_image, win_size=11,data_range=
                       max(image.max()-image.min(),true_image.max()-true_image.min()))


def statistical_results(test_data, model):
    ray_layer = odl_torch.OperatorModule(model.geometry.ray)
    sinos: torch.Tensor = ray_layer(test_data)

    with torch.no_grad():
        recon_img = model.forward(sinos)

    psnr_array=[]
    ssim_array=[]
    

    for i in range(len(test_data)):
        psnr_array.append(psnr(recon_img[i],test_data[i]))
        ssim_array.append(ssim(recon_img[i],test_data[i]))
    
    mean_psnr=np.mean(psnr_array)
    mean_ssim=np.mean(ssim_array)
    mean_square_psnr=np.mean(np.square(psnr_array))
    mean_square_ssim=np.mean(np.square(ssim_array))

    std_psnr=np.sqrt(mean_square_psnr-np.square(mean_psnr))
    std_ssim=np.sqrt(mean_square_ssim-np.square(mean_ssim))

    return np.array([mean_psnr,2*(std_psnr),mean_ssim,2*(std_ssim)])


def test():
    data: torch.Tensor = torch.load("data\kits_phantoms_256.pt").moveaxis(0,1).to(DEVICE)
    data = torch.concat([data[1], data[0], data[2]])
    test_data = data[500:600]
    test_data /= torch.max(torch.max(test_data, dim=-1).values, dim=-1).values[:, None, None]

    geometry = Geometry(1.0, 450, 300)
    model_analytic = RamLak(geometry)

    model_path = "results\gfno_bp-ar1.0-state-dict-450x300.pt"

    # model_fno = GeneralizedFNO_BP.model_from_state_dict(torch.load(model_path, map_location=DEVICE))
    model_mifno = FNOExtrapolatingBP.model_from_state_dict(torch.load("results/BP-fno-fnov1-full-data-ar0.5-450x300.pt", map_location=DEVICE))
    
    print(statistical_results(test_data,model_mifno))

test()