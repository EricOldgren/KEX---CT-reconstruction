import torch
import torch.nn.functional as F

from utils.fno_1d import FNO1d
from utils.polynomials import Legendre, POLYNOMIAL_FAMILY_MAP
from geometries import FBPGeometryBase, DEVICE, DTYPE, CDTYPE

from models.modelbase import FBPModelBase


class FNO_Encoder(FBPModelBase):

    def __init__(self, geometry: FBPGeometryBase, ar: float, M: int, K: int, hidden_layers_sm = [40,40], hidden_layers_pk = [40,40], polynomial_family_key = Legendre.key):
        super().__init__()
        self.geometry = geometry
        self._init_args = (ar, M, K, hidden_layers_sm, hidden_layers_pk, polynomial_family_key)
        self.ar = ar
        self.PolynomialFamily = POLYNOMIAL_FAMILY_MAP[polynomial_family_key]

        channels_s = geometry.projection_size
        channels_phi = geometry.n_known_projections(ar)
        self.fno_pk = FNO1d(channels_phi//2, channels_s, K, layer_widths=hidden_layers_pk, dtype=DTYPE).to(DEVICE)
        self.fno_sm = FNO1d(K//2, channels_phi, M, layer_widths=hidden_layers_sm, dtype=DTYPE).to(DEVICE)
        self.fno_pk_imag = FNO1d(channels_phi//2, channels_s, K, layer_widths=hidden_layers_pk, dtype=DTYPE).to(DEVICE)
        self.fno_sm_imag = FNO1d(K//2, channels_phi, M, layer_widths=hidden_layers_sm, dtype=DTYPE).to(DEVICE)

    def get_init_torch_args(self):
        return self._init_args
    
    def get_extrapolated_sinos(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out = None):
        inp = sinos[known_angles]
        N, Nb, Nu = inp.shape

        out = self.fno_pk(inp.permute(0,2,1)) # shape: N x K x Nb
        out:torch.Tensor = self.fno_sm(out.permute(0,2,1)) # shape: N x M x K
        out_imag = self.fno_pk_imag(inp.permute(0,2,1))
        out_imag:torch.Tensor = self.fno_sm_imag(out_imag.permute(0,2,1))

        return self.geometry.synthesise_series(out+1j*out_imag, self.PolynomialFamily)
    
    def get_extrapolated_filtered_sinos(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out = None):
        return self.geometry.inverse_fourier_transform(self.geometry.fourier_transform(self.get_extrapolated_sinos(sinos, known_angles)*self.geometry.jacobian_det)*self.geometry.ram_lak_filter())
    
    def forward(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out = None):
        return self.geometry.project_backward(self.get_extrapolated_filtered_sinos(sinos, known_angles, angles_out))

    

    
    
