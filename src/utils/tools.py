import torch
from pathlib import Path
import os
from typing import Union

#data and file configs
PathType = Union[os.PathLike, str]
GIT_ROOT = (Path(__file__) / "../../..").resolve()

#Centralized device and dtype for all files. Can be conveniently changed e.g to cpu when debuggging
#These constants should be imported to other files
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float
CDTYPE = torch.cfloat
eps = torch.finfo(DTYPE).eps



#Data loading
def get_htc2022_train_phantoms():
    return torch.stack(torch.load( GIT_ROOT / "data/HTC2022/HTCTrainingPhantoms.pt", map_location=DEVICE)).to(DTYPE)

def get_kits_train_phantoms():
    return torch.load(GIT_ROOT / "data/kits_phantoms_256.pt", map_location=DEVICE)[:500, 1]
#expressions
def no_bdry_linspace(start: float, end: float, n_points: int):
    "linspace with same sampling as the odl default, points at the boundaries are shifted inwards by half of a cell width"
    dx = (end-start)/n_points
    return start + dx/2 + dx*torch.arange(0,n_points, device=DEVICE, dtype=DTYPE)


def MSE(x: torch.Tensor, gt: torch.Tensor):
    "mean squared error function"
    return torch.mean((x-gt)**2)
def RMSE(x: torch.Tensor, gt: torch.Tensor):
    "root mean squared error function"
    return torch.sqrt(MSE(x, gt))

def PSNR(x: torch.Tensor, gt: torch.Tensor):
    "peak signal to noise ratio"
    return 20*torch.log10(torch.max(x))-10*torch.log10(MSE(x,gt))