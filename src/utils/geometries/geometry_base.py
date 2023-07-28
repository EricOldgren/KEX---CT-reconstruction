import torch
from abc import ABC, abstractmethod

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float
CDTYPE = torch.cfloat
eps = torch.finfo(DTYPE).eps

def nearest_power_of_two(n: int):
    P = 1
    while P < n:
        P *= 2
    return P

class FBPGeometryBase(ABC):
    """
    2D Geometry suitable for FBP reconstruction algorithms.
    This includes Parallel beam 2D geometry, Fan Beam equiangular geometry, Flat Fan Beam aquidistant geometry
    """
    @abstractmethod
    def project_forward(self, X: torch.Tensor)->torch.Tensor:
        """Forward projection
        """
    
    @abstractmethod
    def project_backward(self, X: torch.Tensor)->torch.Tensor:
        """Back projection - adjoint of self.project_forward
        """
    
    @abstractmethod
    def fourier_transform(self, sinos: torch.Tensor)->torch.Tensor:
        """Fourier transform alog second argument with appropriate scaling
        """
    
    @abstractmethod
    def inverse_fourier_transform(self, sinohats: torch.Tensor)->torch.Tensor:
        """Inverse of self.fourier_transform
        """
    
    @abstractmethod
    def remlak_filter(self, cutoff_ratio: float = None)->torch.Tensor:
        """Filter used for fbp in fourier domain
        """
    
    @abstractmethod
    def fbp_reconstruct(self, sinos: torch.Tensor)->torch.Tensor:
        """Reconstruct sinos using FBP
        """
    
    @property
    @abstractmethod
    def jacobian_det(self)->torch.Tensor:
        """Jacobian determinant when changing coordinates from geometry sampling parameters to parallel geometry coordinates.
        """