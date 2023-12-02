import torch
import torch.nn as nn
import torch.nn.functional as F
from geometries import FBPGeometryBase, DEVICE, DTYPE, naive_sino_filling
from geometries.extrapolation import RidgeSinoFiller
from models.modelbase import FBPModelBase, load_model_checkpoint
from utils.polynomials import Chebyshev, POLYNOMIAL_FAMILY_MAP
from utils.fno_2d import FNO2d
from typing import Any, List

class FNO_BP2D(FBPModelBase):

    def __init__(self, geometry: FBPGeometryBase, ar: float, hidden_layers: List[int], modes_y = 50, modes_x=50, M = 100, K = 100, PolynomialFamilyKey: int = Chebyshev.key, l2_reg = 0.01):
        super().__init__()
        self._init_args = (ar, hidden_layers, modes_y, modes_x, M, K, PolynomialFamilyKey, l2_reg)
        self.geometry = geometry
        self.ar = ar
        self.M, self.K = M, K
        self.PolynomialFamily = POLYNOMIAL_FAMILY_MAP[PolynomialFamilyKey]
        self.l2_reg = torch.nn.Parameter(torch.tensor(l2_reg), requires_grad=False)

        if modes is None:
            modes = self.geometry.projection_size // 2

        self.known_angles = torch.zeros(geometry.n_projections, device=DEVICE, dtype=torch.bool)
        self.known_angles[:geometry.n_known_projections(ar)] = 1

        self.sinofiller = RidgeSinoFiller(geometry, self.known_angles, M, K, self.PolynomialFamily)
        self.fno2d = FNO2d(modes_y, modes_x, 1, 1, layer_widths=hidden_layers, dtype=DTYPE).to(DEVICE)
    
    def get_init_torch_args(self):
        return self._init_args
    
    def get_extrapolated_sinos(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out = None):
        assert (known_angles == self.known_angles).all(), "rotate sinos so that first known angle is at index 0 to make inference with this model"
        exp = self.sinofiller.forward(sinos, self.l2_reg)
        reflected, known_reg = self.geometry.reflect_fill_sinos(sinos+0, self.known_angles)
        reflected[:, ~known_reg] = exp[:, ~known_reg]
        return reflected
    
    def get_extrapolated_filtered_sinos(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out = None):
        sinos = self.get_extrapolated_sinos(sinos, known_angles)
        return self.fno2d(sinos[:, None])[:, 0]

    def forward(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out = None, use_relu = True):
        if use_relu:
            return F.relu(self.geometry.project_backward(self.get_extrapolated_filtered_sinos(sinos, known_angles)))
        return self.geometry.project_backward(self.get_extrapolated_filtered_sinos(sinos, known_angles))
    