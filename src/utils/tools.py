import torch
from pathlib import Path
import os
from typing import Union, Tuple

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