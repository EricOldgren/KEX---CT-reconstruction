from models.modelbase import ModelBase
from utils.geometry import Geometry, DEVICE
import torch.nn as nn
import torch
import odl.contrib.torch as odl_torch
from typing import List
from models.analyticmodels import ramlak_filter
import torch.nn.functional as F
from utils.smoothing import linear_bandlimited_basis

class FBP(ModelBase):

    def __init__(self, geometry: Geometry, initial_kernel: torch.Tensor = None, trainable_kernel=True, dtype=torch.complex64, **kwargs):
        "Linear layer consisting of a 1D sinogram kernel in frequency domain"
        super().__init__(geometry, **kwargs)
        self.plotkernels = True

        if initial_kernel == None:
            #start_kernel = np.linspace(0, 1.0, geometry.fourier_domain.shape[0]) * np.random.triangular(0, 25, 50)
            #if random.random() < 0.5: start_kernel *= -1
            #self.kernel = nn.Parameter(torch.from_numpy(start_kernel).to(DEVICE), requires_grad=trainable_kernel)
            self.kernel = nn.Parameter(torch.randn(geometry.fourier_domain.shape, dtype=dtype).to(DEVICE), requires_grad=trainable_kernel)
        else:
            assert initial_kernel.shape == geometry.fourier_domain.shape, f"wrong formatted specific kernel {initial_kernel.shape} for geometry {geometry}"
            self.kernel = nn.Parameter(initial_kernel.to(DEVICE, dtype=dtype), requires_grad=trainable_kernel)
    
    def kernels(self) -> List[torch.Tensor]:
        return [self.kernel]

    def forward(self, sinos):
        sino_freq = self.geometry.fourier_transform(sinos)
        filtered_sinos = self.kernel*sino_freq
        filtered_sinos = self.geometry.inverse_fourier_transform(filtered_sinos)

        return F.relu(self.BP_layer(filtered_sinos))
    
    def regularization_term(self):
        "Returns a sum which penalizies large kernel values at large frequencies, in accordance with Nattarer's sampling Theorem"
        penalty_coeffs = torch.zeros(self.geometry.fourier_domain.shape).to(DEVICE) #Create penalty coefficients -- 0 for small frequencies one above Omega
        penalty_coeffs[self.geometry.fourier_domain > self.geometry.omega] = 1.0
        
        (mid_sec, ) = torch.where( (self.geometry.omega*0.9 < self.geometry.fourier_domain) & (self.geometry.fourier_domain <= self.geometry.omega)) # straight line joining free and panalized regions
        penalty_coeffs[mid_sec] = torch.linspace(0, 1.0, mid_sec.shape[0]).to(DEVICE)

        return torch.mean(self.kernel*self.kernel*penalty_coeffs)
    
class GeneralizedFBP(ModelBase):

    def __init__(self, geometry: Geometry, initial_kernel: torch.Tensor = None, trainable_kernel = True, dtype=torch.complex64, **kwargs):
        """
            FBP with a kernel that depends on angle, kernel of shape (phi_size x fourier_shape), initialized with a ramlak filter
        """

        super().__init__(geometry, **kwargs)
        if initial_kernel is not None:
            assert initial_kernel.shape == (geometry.phi_size, geometry.fourier_domain.shape[0]), f"Unexpected shape {initial_kernel.shape}"
            self.kernel = nn.Parameter(initial_kernel.to(DEVICE, dtype=dtype), requires_grad=trainable_kernel)
        else:
            ramlak = ramlak_filter(geometry, dtype)
            self.kernel = nn.Parameter(ramlak[None].repeat(geometry.phi_size, 1), requires_grad=trainable_kernel)
    
    def forward(self, X: torch.Tensor):
        out = self.geometry.fourier_transform(X)
        out = out*self.kernel
        out = self.geometry.inverse_fourier_transform(out)

        return F.relu(self.BP_layer(out))
    
    def regularization_term(self):
        penalty_coeffs = torch.zeros(self.kernel.shape).to(DEVICE) #Create penalty coefficients -- 0 for small frequencies one above Omega
        penalty_coeffs[:, self.geometry.fourier_domain > self.geometry.omega] = 1.0
        
        (mid_sec, ) = torch.where( (self.geometry.omega*0.9 < self.geometry.fourier_domain) & (self.geometry.fourier_domain <= self.geometry.omega)) # straight line joining free and panalized regions
        penalty_coeffs[:, mid_sec] = torch.linspace(0, 1.0, mid_sec.shape[0]).to(DEVICE)

        return torch.mean(self.kernel*self.kernel*penalty_coeffs)

class SmoothFBP(FBP):

    def __init__(self, geometry: Geometry, basis: torch.Tensor = None) -> None:
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
        assert (N,) == geometry.fourier_domain.shape, "basis is wrong length"
        self.basis = basis
        self.coeffs = nn.Parameter(torch.randn((dim, 1)))
    
    @property
    def kernel(self):
        return torch.sum(self.coeffs*self.basis, dim=0)