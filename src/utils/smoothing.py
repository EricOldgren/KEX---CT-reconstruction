import torch
import torch.nn as nn
from utils.parallel_geometry import BasicModel, ParallelGeometry
import odl.contrib.torch as odl_torch

def linear_bandlimited_basis(geometry: ParallelGeometry):
    "Returns two dimensional basis consisting of 1) constant function and 2) linear function"
    beyond_limit = torch.where(geometry.fourier_domain > geometry.omega)
    
    c = torch.ones(geometry.fourier_domain.shape)
    c[beyond_limit] = 0

    x = torch.linspace(0, 1.0, geometry.fourier_domain.shape[0])
    x[beyond_limit] = 0

    return x[None]

#    return torch.stack([c, x])