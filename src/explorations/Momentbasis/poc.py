import torch

from utils.geometry import DEVICE



def get_Un_sink(values: torch.Tensor, n: int, k: int):
    pass
def get_Un_cosk(values: torch.Tensor, n: int, k: int):
    pass



def get_Xn(phis, n):
    """
        Get matrix with columns of trigonometric functions that span the space of linear combinations of hoomogeous polynomials in sin and cos of degree n
        Used to estimate the Fourier coefficients of a moment curve of degree n.
    """
    ONE = torch.ones(phis.shape, dtype=phis.dtype, device=DEVICE)
    if n % 2 == 0:
        return torch.stack(
            [ONE] + [torch.cos(2*k*phis) for k in range(1, n//2+1)] + [torch.sin(2*k*phis) for k in range(1, n//2+1)]
        ).T
    return torch.stack(
            [torch.cos((2*k+1)*phis) for k in range(0, n//2+1)] + [torch.sin((2*k+1)*phis) for k in range(0, n//2+1)]
        ).T

def get_Un(ss, n):
    "Chebyshev polynomial of second kinf, degree n, ss is axis, which must be the interval [-1, 1]"
    return torch.sin((n+1)*torch.acos(ss)) / torch.sqrt(1-ss**2)