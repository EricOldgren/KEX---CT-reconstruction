import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from odl.contrib import torch as odl_torch

from geometries.parallel_geometry.parallel_geometry import ParallelGeometry, DEVICE, missing_range, extend_geometry
from geometries.parallel_geometry.moments import SinoMoments


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

def extrapolate_sinos(sinos: torch.Tensor, known_phis: torch.Tensor, unknown_phis: torch.Tensor, N_moments = 300, ridge_lambda = 0.1):
    """
        Estimates sinogram values in an unknown region based on HLCC using projection onto orthogonal Chebyshev polynomials of the second kind.

        parameters:
            - geometry (Geometry)
            - sinos (Tensor): of shape batch_size x phi_size x t_size
            - unknown_phis (Tensor): angles where the sinograms should be estimated. Of shape (missing_phi_size,)
    """
    N, Np, Nt = sinos.shape
    known_phis = known_phis.to(device=DEVICE, dtype=sinos.dtype)
    unknown_phis = unknown_phis.to(device=DEVICE, dtype=sinos.dtype)
    
    # ts = torch.from_numpy(geometry.translations).to(device=DEVICE, dtype=sinos.dtype)
    # R = geometry.rho #scalle of Chebyshev lengths
    # ss = ts / R #normalized translations in range [-1, 1]
    ss = (1 / (2*Nt) + torch.arange(0,Nt, dtype=sinos.dtype, device=DEVICE)/Nt)
    ds = 2.0 / (Nt-1)
    W = torch.sqrt(1 - ss**2) #Weight function for Chebyshev inner product

    exp = torch.zeros(N, unknown_phis.shape[0], Nt).to(DEVICE, dtype=sinos.dtype)

    for n in range(N_moments):
        #Data
        Xn = get_Xn(known_phis, n)
        Un = torch.sin((n+1)*torch.acos(ss)) / W
        an = torch.sum(sinos*Un, dim=-1) * ds #moment n for every sinogram -- shape: batch_size x phis

        beta = torch.linalg.solve(Xn.T@Xn+ridge_lambda*torch.eye(n+1, device=DEVICE, dtype=sinos.dtype), Xn.T@an.T)
        Xn_exp = get_Xn(unknown_phis, n)
        pred_an = (Xn_exp@beta).T # batch_size x phis

        exp = exp + 2/(torch.pi) * pred_an[:, :, None]*Un*W

    return exp
        


if __name__ == '__main__':
    n_phantoms = 10
    read_data: torch.Tensor = torch.load("data/kits_phantoms_256.pt").moveaxis(0,1).to(DEVICE)
    read_data = torch.concat([read_data[1], read_data[0], read_data[2]])
    read_data = read_data[:n_phantoms]
    read_data /= torch.max(torch.max(read_data, dim=-1).values, dim=-1).values[:, None, None]
    phantoms = read_data

    g = ParallelGeometry(0.5, 450, 300)
    ext_g = extend_geometry(g)
    ray_l = odl_torch.OperatorModule(g.ray)
    full_ray_l = odl_torch.OperatorModule(ext_g.ray)
    sinos = ray_l(phantoms)
    full_sinos = full_ray_l(phantoms)

    fully_projected = extrapolate_sinos(sinos, g.tangles, ext_g.tangles, N_moments=300)
    # exp_sinos = torch.concat([sinos, fully_projected[:, g.phi_size:]], dim=1)

    smp = SinoMoments(ext_g)
    moms = [smp.get_moment(fully_projected, ni) for ni in range(12)]
    proj_moms = [smp.project_moment(mom, ni) for ni, mom in enumerate(moms)]
    mom_diff = np.mean([torch.mean((p_mom-mom)**2).item() for p_mom, mom in zip(proj_moms, moms)])
    print("exp mom MSEs by order:")
    for ni in range(len(moms)):
        print("\t", ni,  torch.mean((proj_moms[ni]-moms[ni])**2).item())
    moms_gt = [smp.get_moment(full_sinos, ni) for ni in range(12)]
    proj_moms_gt = [smp.project_moment(mom, ni) for ni, mom in enumerate(moms_gt)]
    mom_diff_gt = np.mean([torch.mean((mom-p_mom)**2) for mom, p_mom in zip(moms_gt, proj_moms_gt)])
    print("exp mom MSE: ", mom_diff)
    print("mom gt MSE:", mom_diff_gt)

    print("MSE sinos: ", torch.mean((fully_projected-full_sinos)**2))

    plt.subplot(121)
    plt.imshow(full_sinos[0])
    plt.subplot(122)
    plt.imshow(fully_projected[0])
    # plt.imshow(exp_sinos[0])
    plt.show()

    ramlak = RamLak(ext_g)
    recon_full = ramlak(full_sinos[0:10])
    recon_exp = ramlak(fully_projected[0:10])

    print("MSE recon full: ", torch.mean((recon_full[0]-phantoms[0])**2))
    print("MSE recon exp: ", torch.mean((recon_exp[0]-phantoms[0])**2))

    plt.subplot(131)
    plt.imshow(phantoms[0])
    plt.title("GT")
    plt.subplot(132)
    plt.imshow(recon_full[0])
    plt.title("full data")
    plt.subplot(133)
    plt.imshow(recon_exp[0])
    plt.title("exp data")

    plt.show()