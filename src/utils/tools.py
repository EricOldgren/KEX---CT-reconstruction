import torch
from pathlib import Path
import os
from typing import Union, Tuple
import scipy
import numpy as np

#data and file configs
PathType = Union[os.PathLike, str]
GIT_ROOT = (Path(__file__) / "../../..").resolve()

#Centralized device and dtype for all files. Can be conveniently changed e.g to cpu when debuggging
#These constants should be imported to other files
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64 #torch.float
CDTYPE = torch.complex128 #torch.cfloat
eps = torch.finfo(DTYPE).eps

#expressions
def no_bdry_linspace(start: float, end: float, n_points: int):
    "linspace with same sampling as the odl default, points at the boundaries are shifted inwards by half of a cell width"
    dx = (end-start)/n_points
    return start + dx/2 + dx*torch.arange(0,n_points, device=DEVICE, dtype=DTYPE)


def MSE(x: torch.Tensor, gt: torch.Tensor):
    "mean squared error function"
    return torch.mean(torch.abs(x-gt)**2)
def RMSE(x: torch.Tensor, gt: torch.Tensor):
    "root mean squared error function"
    return torch.sqrt(MSE(x, gt))

def PSNR(x: torch.Tensor, gt: torch.Tensor):
    "peak signal to noise ratio"
    return 20*torch.log10(torch.max(torch.abs(x)))-10*torch.log10(MSE(x,gt))

def htc_score(Irs: torch.Tensor, Its: torch.Tensor):
    """Calculate the reconstruction score used in the HTC competetition! Note that the score is returned for each input phantom.

    Citing from HTC score description:

        The score is based on the confusion matrix of the classification of the pixels between empty (0) or material (1),
        it is given by the Matthews correlation coefficient (MCC). The score is betqeen -1 and 1. A score of +1 (best) represents a perfect reconstruction, 0 no better than
        random reconstruction, and −1 (worst) indicates total disagreement between reconstruction and ground truth. 

    Args:
        Irs (torch.Tensor): binary reconstruction tensor of shape (batch_size x 512 x 512)
        Its (torch.Tensor): binary gt phantoms of same shape

    Returns:
        torch.Tensor : score of shape (batch_size)
    """
    assert Irs.dtype == torch.bool and Its.dtype == torch.bool, "input should be in binary format, use appropriate threshholding before calculating the score."

    TPs = torch.sum(Its & Irs, dim=(-1,-2), dtype=torch.float)
    TNs = torch.sum((~Its) & (~Irs), dim=(-1,-2), dtype=torch.float)
    FPs = torch.sum((~Its) & Irs, dim=(-1,-2), dtype=torch.float)
    FNs = torch.sum(Its & (~Irs), dim=(-1,-2), dtype=torch.float)

    # TP = float(len(np.where(AND(It, Ir))[0]))
    # TN = float(len(np.where(AND(NOT(It), NOT(Ir)))[0]))
    # FP = float(len(np.where(AND(NOT(It), Ir))[0]))
    # FN = float(len(np.where(AND(It, NOT(Ir)))[0]))
    # cmat = torch.tensor([[TP, FN], [FP, TN]])
    # Matthews correlation coefficient (MCC)
    
    numerators = TPs * TNs - FPs * FNs
    denominators = torch.sqrt((TPs + FPs) * (TPs + FNs) * (TNs + FPs) * (TNs + FNs))
    res = numerators * 0
    res[denominators != 0] = numerators[denominators != 0] / denominators[denominators != 0]

    return res

def _compute_otsu_criteria(im:torch.Tensor, th):
    """Otsu's method to compute criteria. -- this is from Wikipedia"""
    # create the thresholded image
    h, w = im.shape
    thresholded_im = im * 0
    thresholded_im[im >= th] = 1

    # compute weights
    nb_pixels = im.nelement()
    nb_pixels1 = thresholded_im.count_nonzero()
    weight1 = nb_pixels1 / nb_pixels
    weight0 = 1 - weight1

    # if one of the classes is empty, eg all pixels are below or above the threshold, that threshold will not be considered in the search for the best threshold
    if weight1 == 0 or weight0 == 0:
        return torch.inf

    # find all pixels belonging to each class
    val_pixels1 = im[thresholded_im == 1]
    val_pixels0 = im[thresholded_im == 0]

    # compute variance of these classes
    var1 = torch.var(val_pixels1) if len(val_pixels1) > 1 else 0
    var0 = torch.var(val_pixels0) if len(val_pixels0) > 1 else 0

    res = weight0 * var0 + weight1 * var1
    assert not res.isnan()
    return res # weight0 * var0 + weight1 * var1

def _find_otsu_threshhold(img: torch.Tensor, vals_to_search = 1000):
    # testing all thresholds from 0 to the maximum of the image
    threshold_range = torch.linspace(img.min(), img.max(), vals_to_search)
    # best threshold is the one minimizing the Otsu criteria
    return threshold_range[np.argmin([_compute_otsu_criteria(img, th) for th in threshold_range])]

def segment_imgs(imgs: torch.Tensor, vals_to_search = 1000):
    N, h, w = imgs.shape
    ijs = torch.cartesian_prod(torch.arange(0, 7), torch.arange(0,7)).reshape(7,7,2)
    B = torch.zeros((7,7), dtype=torch.bool)
    B[((ijs-torch.tensor([3, 3]))**2).sum(dim=-1)<=9] = 1
    B = B.numpy()
    res = (imgs*0).to(torch.bool)
    for i in range(N):
        opt_th = _find_otsu_threshhold(imgs[i], vals_to_search)
        # A = (imgs[i] >= opt_th).numpy()
        # res[i] = torch.from_numpy(scipy.ndimage.binary_closing(A)).to(imgs.device)
        res[i] = (imgs[i] >= opt_th)
        

    return res 


#Tensor tools
def pacth_split_image_batch(input: torch.Tensor, patch_shape: Union[int, Tuple[int, int]]):
    """Split batch of 2D tensors into patches of smaller 2d regions

        args: input of shape (... x H x W)
              patch_shape = (py, px) where py | H and px | W

        returns tensor of shape (... x (H/py*W/px) x (px*py)) consists of the pacthes of shape py x px taken in order
    """
    if isinstance(patch_shape, int):
        patch_shape = (patch_shape, patch_shape)
    py, px = patch_shape
    h, w = input.shape[-2:]
    assert h % py == 0, f"incompatible shape {input.shape} with {patch_shape}"
    assert w % px == 0, f"incompatible shape {input.shape} with {patch_shape}"

    return torch.nn.functional.unfold(input[...,None,:,:], patch_shape, padding=0, stride=patch_shape).moveaxis(-1,-2)

def merge_patches(patches: torch.Tensor, img_shape: Tuple[int, int], patch_shape: Union[int, Tuple[int, int]]):
    "inverse of patch_split_image_batch"
    if isinstance(patch_shape, int):
        patch_shape = (patch_shape, patch_shape)
    py, px = patch_shape
    h, w = img_shape
    assert h % py == 0, f"incompatible shape {input.shape} with {patch_shape}"
    assert w % px == 0, f"incompatible shape {input.shape} with {patch_shape}"

    return torch.nn.functional.fold(patches.moveaxis(-1,-2), (h, w), patch_shape, padding=0, stride=patch_shape)[...,0,:,:]