from models.modelbase import ModelBase
from utils.parallel_geometry import ParallelGeometry, DEVICE
import torch.nn as nn
import torch
import odl.contrib.torch as odl_torch
from typing import List
from models.analyticmodels import ramlak_filter
import torch.nn.functional as F
from utils.smoothing import linear_bandlimited_basis
import math

class FBP(ModelBase):

    def __init__(self, geometry: ParallelGeometry, initial_kernel: torch.Tensor = None, use_padding = True, trainable_kernel=True, dtype=torch.complex64, **kwargs):
        "Linear layer consisting of a 1D sinogram kernel in frequency domain"
        super().__init__(geometry, **kwargs)
        self.plotkernels = True
        self.use_padding = use_padding

        omgs = geometry.fourier_domain_padded if use_padding else geometry.fourier_domain
        if initial_kernel == None:
            self.kernel = nn.Parameter(torch.randn(omgs.shape, dtype=dtype).to(DEVICE), requires_grad=trainable_kernel)
        else:
            assert initial_kernel.shape == omgs.shape, f"wrong formatted specific kernel {initial_kernel.shape} for geometry {geometry} with use_padding={use_padding}"
            self.kernel = nn.Parameter(initial_kernel.to(DEVICE, dtype=dtype), requires_grad=trainable_kernel)
    
    def kernels(self) -> List[torch.Tensor]:
        return [self.kernel]

    def forward(self, sinos):
        sino_freq = self.geometry.fourier_transform(sinos, padding=self.use_padding)
        filtered_sinos = self.kernel*sino_freq
        filtered_sinos = self.geometry.inverse_fourier_transform(filtered_sinos, padding=self.use_padding)

        return F.relu(self.BP_layer(filtered_sinos))
    
    def regularization_term(self):
        "Returns a sum which penalizies large kernel values at large frequencies, in accordance with Nattarer's sampling Theorem"
        if self.use_padding==True:
            penalty_coeffs = torch.zeros(self.geometry.fourier_domain_padded.shape).to(DEVICE) #Create penalty coefficients -- 0 for small frequencies one above Omega
            penalty_coeffs[self.geometry.fourier_domain_padded > self.geometry.omega] = 1.0
        
            (mid_sec, ) = torch.where( (self.geometry.omega*0.9 < self.geometry.fourier_domain_padded) & (self.geometry.fourier_domain_padded <= self.geometry.omega)) # straight line joining free and panalized regions
            penalty_coeffs[mid_sec] = torch.linspace(0, 1.0, mid_sec.shape[0]).to(DEVICE)
        
        else:
            penalty_coeffs = torch.zeros(self.geometry.fourier_domain.shape).to(DEVICE) #Create penalty coefficients -- 0 for small frequencies one above Omega
            penalty_coeffs[self.geometry.fourier_domain > self.geometry.omega] = 1.0
        
            (mid_sec, ) = torch.where( (self.geometry.omega*0.9 < self.geometry.fourier_domain) & (self.geometry.fourier_domain <= self.geometry.omega)) # straight line joining free and panalized regions
            penalty_coeffs[mid_sec] = torch.linspace(0, 1.0, mid_sec.shape[0]).to(DEVICE)

        a=self.kernel*self.kernel*penalty_coeffs
        return torch.mean(torch.abs(a.cpu()))

    
class GeneralizedFBP(ModelBase):

    def __init__(self, geometry: ParallelGeometry, initial_kernel: torch.Tensor = None, use_padding = True, trainable_kernel = True, dtype=torch.complex64, **kwargs):
        """
            FBP with a kernel that depends on angle, kernel of shape (phi_size x fourier_shape), initialized with a ramlak filter by default
        """

        super().__init__(geometry, **kwargs)
        self.use_padding = use_padding
        if initial_kernel is not None:
            omgs = geometry.fourier_domain_padded if use_padding else geometry.fourier_domain
            assert initial_kernel.shape == (geometry.phi_size, omgs.shape[0]), f"Unexpected shape {initial_kernel.shape}"
            self.kernel = nn.Parameter(initial_kernel.to(DEVICE, dtype=dtype), requires_grad=trainable_kernel)
        else:
            ramlak = ramlak_filter(geometry, padding=use_padding, cutoff=1.0, dtype=dtype)
            self.kernel = nn.Parameter(ramlak[None].repeat(geometry.phi_size, 1), requires_grad=trainable_kernel)
    
    def forward(self, X: torch.Tensor):
        out = self.geometry.fourier_transform(X, padding=self.use_padding)
        out = out*self.kernel
        out = self.geometry.inverse_fourier_transform(out, padding=self.use_padding)

        return F.relu(self.BP_layer(out))
    
    def regularization_term(self):
        penalty_coeffs = torch.zeros(self.kernel.shape).to(DEVICE) #Create penalty coefficients -- 0 for small frequencies one above Omega
        penalty_coeffs[:, self.geometry.fourier_domain > self.geometry.omega] = 1.0
        
        (mid_sec, ) = torch.where( (self.geometry.omega*0.9 < self.geometry.fourier_domain) & (self.geometry.fourier_domain <= self.geometry.omega)) # straight line joining free and panalized regions
        penalty_coeffs[:, mid_sec] = torch.linspace(0, 1.0, mid_sec.shape[0]).to(DEVICE)

        return torch.mean(torch.abs(self.kernel*self.kernel*penalty_coeffs))

class SmoothFBP(FBP):

    def __init__(self, geometry: ParallelGeometry, basis: torch.Tensor = None, use_padding = True) -> None:
        """
            FBP with kernel as a linear combination of a specified set of basis kernels.

            The basis is given as a 2 rank tensor of shape num_basis_functions x frequency_length
        """
        super(FBP, self).__init__()
        if basis is None:
            basis = linear_bandlimited_basis(geometry)

        self.geometry = geometry
        self.BP_layer = odl_torch.OperatorModule(geometry.BP)

        dim, N = basis.shape
        omgs = geometry.fourier_domain_padded if use_padding else geometry.fourier_domain
        assert (N,) == omgs.shape, "basis is wrong length"
        self.basis = basis
        self.coeffs = nn.Parameter(torch.randn((dim, 1)))
    
    @property
    def kernel(self):
        return torch.sum(self.coeffs*self.basis, dim=0)