from typing import Any
from models.modelbase import FBPModelBase, load_model_from_checkpoint
from geometries import FBPGeometryBase, DEVICE, DTYPE, CDTYPE
import torch.nn as nn
import torch
import torch.nn.functional as F


class AdaptiiveFBP(FBPModelBase):
    """FBP reconstruction method with a trainable filter kernel
    """

    def __init__(self, geometry: FBPGeometryBase, initial_kernel: torch.Tensor = None) -> None:
        super().__init__()
        self._init_args = (initial_kernel,)
        self.geometry = geometry
        if initial_kernel is None:
            initial_kernel = geometry.ram_lak_filter()
        self.kernel = nn.Parameter(initial_kernel.to(DEVICE, dtype=CDTYPE), requires_grad=True)

    def get_init_torch_args(self):
        return self._init_args

    def get_extrapolated_sinos(self, sinos: torch.Tensor):
        "AFBP does no extrapolation, returns input"
        return sinos
    def get_extrapolated_filtered_sinos(self, sinos: torch.Tensor):
        return self.geometry.inverse_fourier_transform(self.geometry.fourier_transform(sinos*self.geometry.jacobian_det)*self.kernel)
    
    def forward(self, sinos: torch.Tensor):
        return F.relu(self.geometry.project_backward(self.get_extrapolated_filtered_sinos(sinos)/2))
    
    @staticmethod
    def load(path):
        return load_model_from_checkpoint(path, AdaptiiveFBP)
    
class FBP(FBPModelBase):
    """Standard fixed FBP model. Not learning.
    """

    def __init__(self, geometry: FBPGeometryBase, kernel: torch.Tensor = None) -> None:
        super().__init__()
        self._init_args = (kernel,)
        self.geometry = geometry
        if kernel is None:
            kernel = geometry.ram_lak_filter()
        self.kernel = nn.Parameter(kernel.to(DEVICE, dtype=CDTYPE), requires_grad=False)

    def get_init_torch_args(self):
        return self._init_args

    def get_extrapolated_sinos(self, sinos: torch.Tensor):
        "FBP does no extrapolation, returns input"
        return sinos
    def get_extrapolated_filtered_sinos(self, sinos: torch.Tensor):
        return self.geometry.inverse_fourier_transform(self.geometry.fourier_transform(sinos*self.geometry.jacobian_det)*self.kernel)
    
    def forward(self, sinos: torch.Tensor):
        return F.relu(self.geometry.project_backward(self.get_extrapolated_filtered_sinos(sinos)/2))
    
    @staticmethod
    def load(path):
        return load_model_from_checkpoint(path, FBP)