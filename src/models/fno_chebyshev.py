import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import odl
import odl.contrib.torch as odl_torch

from utils.parallel_geometry import ParallelGeometry, DEVICE, extend_geometry, missing_range
from utils.inverse_moment_transform import extrapolate_sinos
from models.modelbase import ModelBase
from models.analyticmodels import ramlak_filter
from utils.fno_1d import FNO1d

class FNO_BP_chebyshev(ModelBase):
    """
            FNO-BP using analytic extrapolation of sinograms.

            returns relu(BP(FNO(Xe) + basefilter*Xe)) where: Xe is analytical extrapolation of input, basefilter is given in fourier space, this FNO uses a sliding window to update every row in the extrapolated region 
    """

    def __init__(self, geometry: ParallelGeometry, N_moments = 300, hidden_layers = [40, 20, 10, 1], wrap = 10, stride = 1, basefilter = None, trainable_basefilter = False, **kwargs):
        
        assert geometry.in_middle, "geometry expected to be centered in the middle"
        super().__init__(geometry, **kwargs)
        self.extended_geometry = extend_geometry(geometry)
        self.extended_BP = odl_torch.OperatorModule(self.extended_geometry.BP)
        self.wrap = wrap
        self.stride = stride
        self.N_moments = N_moments
        self.unknown_phis = torch.from_numpy(missing_range(self.geometry, self.extended_geometry)).to(DEVICE, dtype=torch.float)
        self.n_upper = torch.where(self.unknown_phis<self.geometry.tangles[0])[0].shape[0]
        self.n_lower = torch.where(self.unknown_phis>self.geometry.tangles[-1])[0].shape[0]

        modes = torch.where(self.extended_geometry.fourier_domain <= self.extended_geometry.rho)[0].shape[0]
        self.window_down = FNO1d(modes, geometry.phi_size + wrap, stride, layer_widths=hidden_layers, dtype=torch.float).to(DEVICE)
        self.window_up = FNO1d(modes, geometry.phi_size + wrap, stride, layer_widths=hidden_layers, dtype=torch.float).to(DEVICE)

        if basefilter == None:
            basefilter = ramlak_filter(self.extended_geometry, padding=True)
        self.basefilter = nn.Parameter(basefilter, requires_grad=trainable_basefilter)

    def extrapolate(self, X: torch.Tensor):
        "Analytic extrapolation"
        filler = extrapolate_sinos(self.geometry, X, self.unknown_phis, N_moments=self.N_moments)
    
        return F.relu(torch.concat([filler[:, :self.n_upper], X, filler[:, self.n_upper:]], dim=1)) #sinogram is noonnegative
    
    def filter(self, X: torch.Tensor):
        """
            Map the extrapolated sinograms to full angle filtered sinograms. To be used as input for the BP layer.
        """
        filtered_X = self.extended_geometry.inverse_fourier_transform(self.extended_geometry.fourier_transform(X, padding=True)*self.basefilter, padding=True)
        filtered_X = torch.concat([
            torch.flip(filtered_X[:, -self.wrap:], dims=(-1,)),
            filtered_X,
            torch.flip(filtered_X[:, :self.wrap], dims=(-1,))
        ], dim=1)
        upper_edge, lower_edge = self.n_upper+self.wrap, self.n_upper + self.geometry.phi_size+self.wrap
        while lower_edge < self.extended_geometry.phi_size+self.wrap or upper_edge > self.wrap:
            if lower_edge < self.extended_geometry.phi_size+self.wrap:
                filtered_X[:, lower_edge:lower_edge+self.stride] += self.window_down(filtered_X[:, lower_edge-self.geometry.phi_size:lower_edge+self.wrap])
                lower_edge += self.stride
            if upper_edge > self.wrap:
                filtered_X[:, upper_edge-self.stride:upper_edge] += self.window_up(filtered_X[:, upper_edge-self.wrap:upper_edge+self.geometry.phi_size])
                upper_edge -= self.stride
        
        return filtered_X[:, self.n_upper:-self.n_lower]
    
    def backproject(self, X: torch.Tensor):
        return F.relu(self.extended_BP(X))

    def forward(self, X: torch.Tensor):
        out = self.extrapolate(X)
        out = self.filter(out)
        return self.backproject(out)

    