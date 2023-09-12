import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
 
from utils.polynomials import Legendre, POLYNOMIAL_FAMILY_MAP
from geometries import FBPGeometryBase, DEVICE, DTYPE, CDTYPE, get_moment_mask
from models.modelbase import FBPModelBase, load_model_checkpoint, PathType 


class Series_BP(FBPModelBase):

    def __init__(self, geometry: FBPGeometryBase, ar: float, M: int, K: int, polynomial_family_key: int = Legendre.key):
        assert 0 < ar <= 1.0, f"angle ratio, {ar} is invalid"
        super().__init__()
        self.geometry = geometry
        self._init_args = (ar, M, K, polynomial_family_key)

        n_known_angles = geometry.n_known_projections(ar)
        self.M, self.K = M, K
        self.PolynomialFamily = POLYNOMIAL_FAMILY_MAP[polynomial_family_key]

        h, w, c = n_known_angles, geometry.projection_size, 1
        next_c = lambda c : 8 if c == 1 else min(64, c*2)
        conv_layers = []
        while min(h, w) >= 4:
            conv_layers.append(nn.Conv2d(c, next_c(c), (4,4), 2, padding=0, device=DEVICE))
            # conv_layers.append(nn.LeakyReLU(0.2))
            c = next_c(c)
            h = (h-4)//2 + 1
            w = (w-4)//2 + 1
 
        self.moment_mask = get_moment_mask(torch.zeros((1,M,K), device=DEVICE))
        n_coeffs = self.moment_mask.count_nonzero()
        self.conv_layers = nn.ModuleList(conv_layers)
        self.lin_out = nn.Linear(64, n_coeffs, dtype=CDTYPE, device=DEVICE)
    
    def get_init_torch_args(self):
        return self._init_args

    def get_extrapolated_sinos(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out: torch.Tensor = None):
        out = sinos[:,None, known_angles]
        N, h, w = sinos.shape
        for i, conv in enumerate(self.conv_layers):
            out = conv(out)
            out = F.leaky_relu(out, 0.2)

        out = torch.mean(out, dim=(-1,-2), dtype=CDTYPE)

        out = self.lin_out(out)
        coefficients = torch.zeros((N, self.M, self.K), device=DEVICE, dtype=CDTYPE)
        coefficients[:, self.moment_mask] += out

        return self.geometry.synthesise_series(coefficients, self.PolynomialFamily)
    
    def get_extrapolated_filtered_sinos(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out: torch.Tensor = None):
        sinos = self.get_extrapolated_sinos(sinos, known_angles, angles_out)
        return self.geometry.inverse_fourier_transform(self.geometry.fourier_transform(sinos*self.geometry.jacobian_det)*self.geometry.ram_lak_filter()/2)
    
    def forward(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out: torch.Tensor = None):
        return self.geometry.project_backward(self.get_extrapolated_filtered_sinos(sinos, known_angles, angles_out))
    
    @staticmethod
    def load(path: PathType):
        return load_model_checkpoint(path, Series_BP).model
    

        
if __name__ == "__main__":
    from utils.tools import MSE, htc_score
    from utils.polynomials import Legendre, Chebyshev
    from utils.data import get_htc2022_train_phantoms
    from geometries import HTC2022_GEOMETRY
    import matplotlib
    import matplotlib.pyplot as plt
    ar = 0.25
    geometry = HTC2022_GEOMETRY
    PHANTOMS = get_htc2022_train_phantoms()
    SINOS = geometry.project_forward(PHANTOMS)
    M, K = 120, 60

    model = Series_BP(geometry, ar, M, K, Legendre.key)

    print(model)
    sinos_la, knwon_angles = geometry.zero_cropp_sinos(SINOS, ar, 0)

    out = model.forward(sinos_la, knwon_angles)  

    print(MSE(out, PHANTOMS))


