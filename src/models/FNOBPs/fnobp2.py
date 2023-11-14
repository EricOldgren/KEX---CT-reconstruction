import torch
from utils.tools import DTYPE, DEVICE
from utils.fno_1d import SpectralConv1d
from utils.polynomials import Chebyshev, POLYNOMIAL_FAMILY_MAP

from geometries import FBPGeometryBase
from geometries.extrapolation import RidgeSinoFiller
from models.modelbase import FBPModelBase

from typing import List

class SpectralDance(torch.nn.Module):

    def __init__(self):
        super().__init__()

        modes = 280
        self.activation = torch.nn.LeakyReLU(0.2)
        self.spectral_inp = torch.nn.ModuleList([SpectralConv1d(60, 60, max_mode=modes, dtype=DTYPE).to(DEVICE) for _ in range(12)])
        self.spectral1 = torch.nn.ModuleList([SpectralConv1d(60, 15, modes, DTYPE).to(DEVICE) for _ in range(12)])
        self.spectral2 = torch.nn.ModuleList([SpectralConv1d(60, 20, modes, DTYPE).to(DEVICE) for _ in range(3)])

        self.despectral1 = torch.nn.ModuleList([SpectralConv1d(60, 180, modes, DTYPE).to(DEVICE)])
        self.despectral2 = torch.nn.ModuleList(([SpectralConv1d(30, 60, modes, DTYPE) for _ in range(12)]))

        self.out = torch.nn.ModuleList([torch.nn.Conv1d(120, 60, 1) for _ in range(12)])

    def forward(self, X: torch.Tensor):
        N, Nb, Nu = X.shape
        assert Nb == 720

        X = X.reshape(N, 12, 60, Nu)
        out_top = X*0
        for i in range(12):
            out_top[:, i] = self.activation(self.spectral_inp[i](X[:, i]))
        
        #Encoding
        out_level1 = X[:, :, :15] * 0
        for i in range(12):
            out_level1[:, i] = self.activation(self.spectral1[i](out_top[:, i]))
        in_level2 = out_level1.reshape(N, 3, 60, Nu)
        out_level2 = in_level2[:, :, :20] * 0
        for i in range(3):
            out_level2[:, i] = self.activation(self.spectral2[i](in_level2[:, i]))

        #Decoding
        in_delevel1 = out_level2.reshape(N, 1, 60, Nu)
        out_delevel1 = self.activation(self.despectral1[0](in_delevel1[:, 0]))[:, None]
        out_delevel1 = out_delevel1.reshape(N, 12, 15, Nu)
        in_delevel2 = torch.concat([
            out_level1, out_delevel1
        ], dim=2)
        assert in_delevel2.shape == (N, 12, 30, Nu)
        out_delevel2 = out_top*0
        for i in range(12):
            out_delevel2[:, i] = self.activation(self.despectral2[i](in_delevel2[:, i]))
        out = torch.concat([
            out_top, out_delevel2
        ], dim=2)
        assert out.shape == (N, 12, 120, Nu)

        #Feed forward
        res = X*0
        for i in range(12):
            res[:, i] = self.out[i](out[:, i])

        return res.reshape(N, Nb, Nu)

class FNOBP2(FBPModelBase):

    def __init__(self, geometry: FBPGeometryBase, ar: float, M = 50, K = 50, PolynomialFamilyKey: int = Chebyshev.key, l2_reg = 0.01):
        super().__init__()
        self._init_args = (ar, M, K, PolynomialFamilyKey, l2_reg)
        self.geometry = geometry
        self.ar = ar
        self.M, self.K = M, K
        self.PolynomialFamily = POLYNOMIAL_FAMILY_MAP[PolynomialFamilyKey]
        self.l2_reg = torch.nn.Parameter(torch.tensor(l2_reg), requires_grad=False)

        self.known_angles = torch.zeros(geometry.n_projections, device=DEVICE, dtype=torch.bool)
        self.known_angles[:geometry.n_known_projections(ar)] = 1

        self.sinofiller = RidgeSinoFiller(geometry, self.known_angles, M, K, self.PolynomialFamily)
        self.dance = SpectralDance()
    
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
        return self.dance(sinos)

    def forward(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out = None, use_relu = True):
        if use_relu:
            return torch.nn.functional.relu(self.geometry.project_backward(self.get_extrapolated_filtered_sinos(sinos, known_angles)))
        return self.geometry.project_backward(self.get_extrapolated_filtered_sinos(sinos, known_angles))


if __name__ == "__main__":
    from geometries.data import HTC2022_GEOMETRY, get_htc_traindata
    import time
    geometry = HTC2022_GEOMETRY
    sinos, phantoms = get_htc_traindata()

    model = SpectralDance()
    T = 30
    start = time.time()
    for _ in range(T):
        out = model.forward(sinos)
    model_time = (time.time()-start)/T
    print(model_time)


        


