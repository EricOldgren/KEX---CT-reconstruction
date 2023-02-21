from .geometry import BasicModel, Geometry, DEVICE
import torch
from math import ceil

def analytic_model(geometry: Geometry):

    T_min, T_max = geometry.detector_partition.min_pt[0], geometry.detector_partition.max_pt[0]
    D = T_max - T_min
    dw = 1 / D
    kernel = torch.arange(0, ceil(0.5 + 0.5*geometry.t_size)).to(DEVICE)*dw
    
    return BasicModel(geometry, kernel, trainable_kernel=False)

