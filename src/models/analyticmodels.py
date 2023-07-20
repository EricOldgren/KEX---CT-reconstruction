from utils.parallel_geometry import ParallelGeometry, DEVICE
from models.modelbase import ModelBase
from odl.contrib import torch as odl_torch
import torch
import numpy as np
from math import ceil
import torch.nn.functional as F

def ramlak_filter(geometry: ParallelGeometry, padding=False, cutoff: float = None, dtype=torch.complex64):
    """
        Constructs a Ram-Lak filter in fourier domain.

        Parameters:
            - geometry (Geometry): the geometry to reconstruct in
            - padding (bool): wether to use padding
            - cutoff (float | None) : ratio of max frequency to cutoff filter from. If set to None (default) the cutof frequency from Nattarer's sampling Theorem is used.
    """
    omgs = geometry.fourier_domain if padding == False else geometry.fourier_domain_padded
    kernel = omgs /  (2*np.pi)
    if cutoff is None:
        kernel[omgs > geometry.omega * 1.0] = 0
    else:
        kernel[omgs > omgs[-1] * cutoff] = 0
    return kernel.to(device=DEVICE, dtype=dtype)


class RamLak(ModelBase):

    def __init__(self, geometry: ParallelGeometry, dtype=torch.complex64, use_padding = True, cutoff = 1.0, **kwargs):
        "FBP based on Radon's invesrion formula. Uses a |omega| filter with a hard cut."
        super().__init__(geometry, **kwargs)
        self.plotkernels = True
        
        self.kernel = torch.nn.Parameter(ramlak_filter(geometry, padding=use_padding, cutoff=cutoff, dtype=dtype), requires_grad=False)
        self.use_padding = use_padding

    def kernels(self) -> 'list[torch.Tensor]':
        return [self.kernel]

    def forward(self, sinos):
        sino_freq = self.geometry.fourier_transform(sinos, padding=self.use_padding)
        filtered_sinos = self.kernel*sino_freq
        filtered_sinos = self.geometry.inverse_fourier_transform(filtered_sinos, padding=self.use_padding)

        return self.BP_layer(filtered_sinos)

    def return_filtered_sino(self,sinos):
        sino_freq = self.geometry.fourier_transform(sinos, padding=self.use_padding)
        filtered_sinos = self.kernel*sino_freq
        filtered_sinos = self.geometry.inverse_fourier_transform(filtered_sinos, padding=self.use_padding)

        return filtered_sinos