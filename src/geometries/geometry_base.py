import torch
from abc import ABC, abstractmethod

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float
CDTYPE = torch.cfloat
eps = torch.finfo(DTYPE).eps

def next_power_of_two(n: int):
    P = 1
    while P < n:
        P *= 2
    return P

class FBPGeometryBase(torch.nn.Module, ABC):
    """
    2D Geometry suitable for FBP reconstruction algorithms.
    This includes Parallel beam 2D geometry, Fan Beam equiangular geometry, Flat Fan Beam aquidistant geometry
    """
    jacobian_det: torch.Tensor
    "Jacobian determinant when changing coordinates from geometry sampling parameters to parallel geometry coordinates."
    ws: torch.Tensor
    "Frequencies where the DFT is sampled at"

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
    def ram_lak_filter(self, cutoff_ratio: float = None, full_size = False)->torch.Tensor:
        """Ram-Lak Filter used for fbp. Filter is given in fourier domain.

            Args:
                - cutoff_ratio: ratio of max frequency to cutoff filter from
                - full_size: if True filter is the same shape as sinograms in geometry, else it is 1 x Nt - where Nt is size along last axis of sinograms
        """
    
    @abstractmethod
    def fbp_reconstruct(self, sinos: torch.Tensor)->torch.Tensor:
        """Reconstruct sinos using FBP
        """
    
    @property
    @abstractmethod
    def n_projections(self):
        "number of projections - height of sinograms"
    
    @property
    @abstractmethod
    def projection_size(self):
        "number of samples per projection - length of row in sinogram"
        