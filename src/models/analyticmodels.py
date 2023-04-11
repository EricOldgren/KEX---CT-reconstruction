from utils.geometry import Geometry, DEVICE
from models.modelbase import ModelBase
from odl.contrib import torch as odl_torch
import torch
import numpy as np
from math import ceil
import torch.nn.functional as F

def ramlak_filter(geometry: Geometry, dtype=torch.complex64):
    kernel = geometry.fourier_domain /  (2*np.pi)
    kernel[geometry.fourier_domain > geometry.omega * 1.0] = 0
    return kernel.to(dtype=dtype)


class RamLak(ModelBase):

    def __init__(self, geometry: Geometry, dtype=torch.float32, **kwargs):
        "FBP based on Radon's invesrion formula and Nattarer's sampling theorem. Uses a |x| filter with a hard cut"
        super().__init__(geometry, **kwargs)
        self.plotkernels = True
        
        self.kernel = torch.nn.Parameter(ramlak_filter(geometry, dtype), requires_grad=False)

    def kernels(self) -> 'list[torch.Tensor]':
        return [self.kernel]

    def forward(self, sinos):
        sino_freq = self.geometry.fourier_transform(sinos)
        filtered_sinos = self.kernel*sino_freq
        filtered_sinos = self.geometry.inverse_fourier_transform(filtered_sinos)

        return F.relu(self.BP_layer(filtered_sinos))