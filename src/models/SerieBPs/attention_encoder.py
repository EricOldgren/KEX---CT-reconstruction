import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from utils.polynomials import Legendre, POLYNOMIAL_FAMILY_MAP
from geometries import FBPGeometryBase, CDTYPE, DEVICE, enforce_moment_constraints
from models.modelbase import FBPModelBase


MODEL_DIM = 512


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    d = Q.shape[-1]
    activations = F.softmax(Q@K.T / np.sqrt(d), dim=-1)
    return activations@V

class AttentionEncoder(FBPModelBase):

    def __init__(self, geometry: FBPGeometryBase, ar: float, M: int, K: int, n_keyvals: int, PolynomialFamilyKey = Legendre.key, strict_moments = True):
        super().__init__()
        self._init_args = (geometry, ar, n_keyvals, strict_moments)
        self.geometry = geometry
        self.ar = ar
        self.PolynomialFamily = POLYNOMIAL_FAMILY_MAP[PolynomialFamilyKey]
        self.strict_moments = strict_moments

        in_dim = geometry.n_known_projections(ar)*geometry.projection_size
        out_dim = MODEL_DIM

        self.keys = nn.Parameter(torch.randn((n_keyvals, in_dim), dtype=CDTYPE, device=DEVICE)/(in_dim), requires_grad=True)
        self.vals = nn.Parameter(torch.randn((n_keyvals, out_dim), dtype=CDTYPE, device=DEVICE)/(out_dim), requires_grad=True)

        self.ffd = nn.Linear(out_dim, M*K)

    def get_init_torch_args(self):
        return self._init_args
    
    def get_extrapolated_sinos(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out = None):
        inp = sinos[:, known_angles]

        out = scaled_dot_product_attention(inp, self.keys, self.vals)
        coefficients = self.ffd(out)
        if self.strict_moments:
            enforce_moment_constraints(coefficients)

        return self.geometry.synthesise_series(coefficients, self.PolynomialFamily)
    
    def get_extrapolated_filtered_sinos(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out = None):
        return self.geometry.inverse_fourier_transform(self.geometry.fourier_transform(self.get_extrapolated_sinos(sinos, known_angles, angles_out)*self.geometry.jacobian_det)*self.geometry.ram_lak_filter())
    
    def forward(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out = None):
        return self.geometry.fbp_reconstruct(self.get_extrapolated_sinos(sinos, known_angles, angles_out))