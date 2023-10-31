from typing import Any
from utils.polynomials import Chebyshev, POLYNOMIAL_FAMILY_MAP
from models.modelbase import FBPModelBase, load_model_checkpoint
from geometries import FBPGeometryBase, DEVICE, DTYPE, CDTYPE
from geometries.extrapolation import RidgeSinoFiller
import torch.nn as nn
import torch
import torch.nn.functional as F


class AdaptiveFBP(FBPModelBase):
    """FBP reconstruction method with a trainable filter kernel
    """

    def __init__(self, geometry: FBPGeometryBase, ar:float, initial_kernel: torch.Tensor = None, M = 50, K = 50, polynomialFamilyKey=Chebyshev.key, l2_reg=0.01) -> None:
        super().__init__()
        self._init_args = (ar, initial_kernel, M, K, polynomialFamilyKey)
        self.geometry = geometry
        self.PolynomialFamily = POLYNOMIAL_FAMILY_MAP[polynomialFamilyKey]
        self.l2_reg = torch.nn.Parameter(torch.tensor(l2_reg), requires_grad=False)
        if initial_kernel is None:
            initial_kernel = geometry.ram_lak_filter()
        self.kernel = nn.Parameter(initial_kernel.to(DEVICE, dtype=CDTYPE), requires_grad=True)

        self.known_angles = torch.zeros(geometry.n_projections, device=DEVICE, dtype=torch.bool)
        self.known_angles[:geometry.n_known_projections(ar)] = 1

        self.sinofiller = RidgeSinoFiller(geometry, self.known_angles, M, K, self.PolynomialFamily)


    def get_init_torch_args(self):
        return self._init_args

    def get_extrapolated_sinos(self, sinos: torch.Tensor, known_angles: torch.Tensor, out_angles: torch.Tensor = None):
        "AFBP does no extrapolation, returns input"
        assert (known_angles == self.known_angles).all(), "rotate sinos so that first known angle is at index 0 to make inference with this model"
        exp = self.sinofiller.forward(sinos, self.l2_reg)
        reflected, known_reg = self.geometry.reflect_fill_sinos(sinos+0, self.known_angles)
        reflected[:, ~known_reg] = exp[:, ~known_reg]
        return reflected

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