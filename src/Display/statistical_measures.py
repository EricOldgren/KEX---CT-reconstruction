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
from skimage.metrics import structural_similarity as ssim_import



def MSE(img, true_img):
    return torch.mean((img-true_img)*(img-true_img))

def psnr(img,true_img):
    return 20*math.log10(torch.max(true_img))-10*math.log10(MSE(img,true_img))

def ssim(image, true_image):
    image=image.detach().numpy()
    true_image=true_image.detach().numpy()
    return ssim_import(image, true_image, win_size=11,data_range=
                       max(image.max()-image.min(),true_image.max()-true_image.min()))

