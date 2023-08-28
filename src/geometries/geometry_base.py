import torch
from abc import ABC, abstractmethod
from typing import Tuple

from utils.tools import DEVICE, DTYPE, CDTYPE, eps

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
    
    @abstractmethod
    def zero_cropp_sinos(self, sinos: torch.Tensor, ar: float, start_ind: int)->Tuple[torch.Tensor, torch.Tensor]:
        """
            Cropp sinograms to limited angle data. Sinos are set to zero outside cropped region

            return cropped_sinos, known_beta_bool
        """

    @abstractmethod
    def reflect_fill_sinos(self, sinos: torch.Tensor, known_beta_bools: torch.Tensor, linear_interpolation = False)->Tuple[torch.Tensor, torch.Tensor]:
        """
            In place flling of limited angle sinograms
            applied on full 360deg sinograms, fills unknown region of sinogram by finding equivalent lines on opposite side

            return: filled_sinos, new_known_region
        """
    
    @abstractmethod
    def get_init_args(self):
        """Get args used in init method. Necessary to reload geometry after saving a model.
        """
    
    @property
    @abstractmethod
    def n_projections(self):
        "number of projections - height of sinograms"
    
    @property
    @abstractmethod
    def projection_size(self):
        "number of samples per projection - length of row in sinogram"


def naive_sino_filling(sinos: torch.Tensor, known_beta_bools: torch.Tensor):
    "Interpolate limited angle sinogram linearly along angle direction. This is only intended as a first input to a network."
    N, n_projections, projection_size = sinos.shape
    all_inds = torch.arange(0, n_projections)
    known_inds, unknown_inds = all_inds[known_beta_bools], all_inds[~known_beta_bools]
    n_unknown = known_beta_bools.count_nonzero().item()

    res = sinos + 0
    lower_ind = known_inds[-1]
    last_vals = sinos[:, lower_ind:lower_ind+1, :]
    for i in range(n_unknown):
        upper_ind = known_inds[i]
        vals = sinos[:, upper_ind:upper_ind+1, :]
        if i == 0: #wrapping point
            between = (unknown_inds < upper_ind) | (unknown_inds >= lower_ind)
        else:
            between = (unknown_inds < upper_ind) & (unknown_inds >= lower_ind)
        between = unknown_inds[between]
        nb = between.nelement()
        res[:, between, :] = last_vals + (vals-last_vals)*(torch.arange(1, nb+1, device=sinos.device) / (nb+1))[None, :, None] 

        lower_ind = upper_ind
        last_vals = vals

    return res