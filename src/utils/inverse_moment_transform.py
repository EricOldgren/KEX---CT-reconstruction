import torch
import numpy as np
import sklearn
from odl.contrib import torch as odl_torch

from utils.geometry import Geometry, DEVICE


def extrapolate_sinos(geometry: Geometry, sinos: torch.Tensor, N_moments = 50):
    """
        Extrapolate sinogram based on HLCC using projection onto orthogonal Chebyshev polynomials of the second kind
    """
    Np, Nt = geometry.phi_size, geometry.t_size
    ts = torch.from_numpy(geometry.translations).to(device=DEVICE, dtype=sinos.dtype)
    phis = torch.from_numpy(geometry.angles).to(device=DEVICE, dtype=sinos.dtype)
    R = geometry.rho
    ss = ts / R
    W = torch.sqrt(1 - ss**2)

    ONE = torch.ones((Np,), dtype=sinos.dtype, device=DEVICE)

    for n in range(N_moments):

        Un = torch.sin((n+1)*torch.acos(ss)) / W
        an = torch.sum(sinos*Un, dim=-1) #moment n for every sinogram
        if n % 2 == 0:
            Xn = torch.stack(
                [ONE] + [torch.cos(2*k) for k in range(1, n//2+1)] + [torch.sin(2*k) for k in range(1, n//2+1)]
            )
        else:
            Xn = torch.stack(
                [torch.cos(2*k+1) for k in range(0, n//2+1)] + [torch.sin(2*k+1) for k in range(0, n//2+1)]
            )
        #Ridge regression to determine trigonometric coefficients
        


if __name__ == '__main__':
    n_phantoms = 2
    read_data: torch.Tensor = torch.load("data/kits_phantoms_256.pt").moveaxis(0,1).to(DEVICE)
    read_data = torch.concat([read_data[1], read_data[0], read_data[2]])
    read_data = read_data[:n_phantoms] # -- uncomment to read this data
    read_data /= torch.max(torch.max(read_data, dim=-1).values, dim=-1).values[:, None, None]
    phantoms = read_data

    g = Geometry(0.5, 450, 300)
    ray_l = odl_torch.OperatorModule(g.ray)
    sinos = ray_l(phantoms)

    extrapolate_sinos(g, sinos)