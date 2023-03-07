from .geometry import BasicModel, Geometry, DEVICE
import torch
import numpy as np
from math import ceil

def analytic_model(geometry: Geometry):

    T_min, T_max = geometry.detector_partition.min_pt[0], geometry.detector_partition.max_pt[0]
    D = T_max - T_min
    dw = 2*np.pi / D #This is one step in the fft transformed sinogram space -- maybe 2pi have to go back :/
    O = geometry.omega #Maximum bandwidth that can be reconstructed exactly using sampling theorem
    straigh_line = torch.arange(0, ceil(0.5 + 0.5*geometry.t_size)).to(DEVICE)*dw
    
    straigh_line[straigh_line > O] = 0.0
    kernel = straigh_line / 2 / np.pi

    res =  BasicModel(geometry, kernel, trainable_kernel=False)
    return res
