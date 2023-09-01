import torch
import torch.nn.functional as F

from typing import Tuple

from utils.tools import DEVICE, DTYPE
from utils.polynomials import POLYNOMIAL_FAMILY_MAP, Legendre
from utils.fno_1d import FNO1d
from geometries import FBPGeometryBase
from models.modelbase import FBPModelBase
from models.FNOBPs.fnobp import FNO_BP


class IterativePjectingFNO(FBPModelBase):
    """
        This implements iteravtive reconstruction process
        input X
        do 1,...,n_iters
            X = relu(restore(project(FNO_i(X))))
        return relu(FBP(X))
    """

    def __init__(self, geometry: FBPGeometryBase, n_iters: int, n_degs: int, fno_layer_widths: Tuple[Tuple[int]], polynomial_family_key: int = Legendre.key) -> None:
        super().__init__()
        self.geometry = geometry
        self._init_args = (n_iters, n_degs, fno_layer_widths, polynomial_family_key)
        self.polynomial_family = POLYNOMIAL_FAMILY_MAP[polynomial_family_key]

        self.n_degs = n_degs
        self.n_iters = n_iters

        self.fnos = torch.nn.ModuleList(
            FNO1d(geometry.projection_size//2, geometry.n_projections, geometry.n_projections, fno_layer_widths[i], dtype=DTYPE).to(DEVICE) for i in range(n_iters)
        )
    
    def get_init_torch_args(self):
        return self._init_args

    def get_extrapolated_sinos(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out: torch.Tensor):
        res = sinos + 0
        for fno in range(self.fnos):
            res[:, ~known_angles] = F.relu(self.geometry.moment_project(fno(sinos), self.polynomial_family, self.n_degs))

        return res
    
    def get_extrapolated_filtered_sinos(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out: torch.Tensor):
        return self.geometry.inverse_fourier_transform(self.geometry.fourier_transform(sinos*self.geometry.jacobian_det)*self.geometry.ram_lak_filter()/2)
    
    def forward(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out: torch.Tensor):
        return F.relu(self.geometry.project_backward(self.get_extrapolated_filtered_sinos(sinos, known_angles, angles_out)))