import torch

#
#Centralized device and dtype for all files. Can be conveniently changed e.g to cpu when debuggging
#
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float
CDTYPE = torch.cfloat
eps = torch.finfo(DTYPE).eps



#Tools
def no_bdry_linspace(start: float, end: float, n_points: int):
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