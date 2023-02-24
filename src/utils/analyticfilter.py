from .geometry import BasicModel, Geometry, DEVICE
import torch
import numpy as np
from math import ceil

def analytic_model(geometry: Geometry):

    T_min, T_max = geometry.detector_partition.min_pt[0], geometry.detector_partition.max_pt[0]
    D = T_max - T_min
    dw = 1 / D
    O = np.pi * min(1 / (geometry.dphi*geometry.rho), 2 / geometry.dt)
    kernel = torch.arange(0, ceil(0.5 + 0.5*geometry.t_size)).to(DEVICE)*dw
    kernel[kernel > O] = 0.0
    
    return BasicModel(geometry, kernel, trainable_kernel=False)

