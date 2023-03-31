import torch
import torch.nn as nn
from utils.geometry import BasicModel, Geometry
import odl.contrib.torch as odl_torch

def linear_bandlimited_basis(geometry: Geometry):
    "Returns two dimensional basis consisting of 1) constant function and 2) linear function"
    beyond_limit = torch.where(geometry.fourier_domain > geometry.omega)
    
    c = torch.ones(geometry.fourier_domain.shape)
    c[beyond_limit] = 0

    x = torch.linspace(0, 1.0, geometry.fourier_domain.shape[0])
    x[beyond_limit] = 0

    return x[None]

#    return torch.stack([c, x])


class SmoothedModel(BasicModel):

    def __init__(self, geometry: Geometry, basis: torch.Tensor = None) -> None:
        """
            FBP with kernel as a linear combination of a specified set of basis kernels.

            The basis is given as a 2 rank tensor of shape num_basis_functions x frequency_length
        """
        super(BasicModel, self).__init__()
        if basis is None:
            basis = linear_bandlimited_basis(geometry)

        self.geometry = geometry
        self.BP_layer = odl_torch.OperatorModule(geometry.BP)

        dim, N = basis.shape
        assert (N,) == geometry.fourier_domain.shape, "basis is wrong length"
        self.basis = basis
        self.coeffs = nn.Parameter(torch.randn((dim, 1)))
    
    @property
    def kernel(self):
        return torch.sum(self.coeffs*self.basis, dim=0)