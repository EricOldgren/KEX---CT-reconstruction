from typing import Any
from models.modelbase import FBPModelBase, load_model_checkpoint
from geometries import FBPGeometryBase, DEVICE, DTYPE, CDTYPE
import torch.nn as nn
import torch
import torch.nn.functional as F


class AdaptiveFBP(FBPModelBase):
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

    def get_extrapolated_sinos(self, sinos: torch.Tensor, known_angles: torch.Tensor, out_angles: torch.Tensor = None):
        "AFBP does no extrapolation, returns input"
        res = sinos + 0
        self.geometry.reflect_fill_sinos(res, known_angles)
        return res

    def get_extrapolated_filtered_sinos(self, sinos: torch.Tensor, known_angles: torch.Tensor, out_angles: torch.Tensor = None):
        "only first argument used"
        return self.geometry.inverse_fourier_transform(self.geometry.fourier_transform(self.get_extrapolated_sinos(sinos, known_angles, out_angles)*self.geometry.jacobian_det)*self.kernel)
        
    
    def forward(self, sinos: torch.Tensor, known_angles: torch.Tensor, out_angles: torch.Tensor = None):
        return F.relu(self.geometry.project_backward(self.get_extrapolated_filtered_sinos(sinos, known_angles, out_angles)/2))
    
    @staticmethod
    def load_checkpoint(path):
        return load_model_checkpoint(path, AdaptiveFBP)
    
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

    def get_extrapolated_sinos(self, sinos: torch.Tensor, known_angles: torch.Tensor, out_angles: torch.Tensor = None):
        "FBP does no extrapolation, returns input"
        res = sinos + 0
        self.geometry.reflect_fill_sinos(res, known_angles)
        return res
    def get_extrapolated_filtered_sinos(self, sinos: torch.Tensor, known_angles: torch.Tensor, out_angles: torch.Tensor = None):
        return self.geometry.inverse_fourier_transform(self.geometry.fourier_transform(self.get_extrapolated_sinos(sinos, known_angles, out_angles)*self.geometry.jacobian_det)*self.kernel)
         
    def forward(self, sinos: torch.Tensor, known_angles: torch.Tensor, out_angles: torch.Tensor):
        return F.relu(self.geometry.project_backward(self.get_extrapolated_filtered_sinos(sinos, known_angles, out_angles)/2))
    
    @staticmethod
    def load_checkpoint(path):
        return load_model_checkpoint(path, FBP)