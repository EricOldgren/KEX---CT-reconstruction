import torch

from utils.geometry import Geometry, setup, DEVICE, extend_geometry
import scipy
from math import ceil
import torch.nn as nn

ZERO_THRESHOLD = 1e-5

class SinoMoments:

    def __init__(self, full_geometry: Geometry, n_moments = 12) -> None:
        self.geometry = full_geometry
        assert full_geometry.ar == 1.0, "assumng full geometry"
        self.n_moments = n_moments

        phis = torch.linspace(0.0, torch.pi, steps=full_geometry.phi_size, device=DEVICE)
        self.natural_basis = []
        "list of rank 2 tensors, each a basis spanning the subspace in which the n:th moment should lie in. Basis listed rowwise"
        self.on_basis = []
        "list of ON bases, each basis rowwise"
        for ni in range(n_moments):
            #construct basis for homogenous polynomials of degree n in sin and cos
            bi = torch.zeros(ni+1, full_geometry.phi_size, device=DEVICE)
            for k in range(0, ni+1):
                bik = torch.cos(phis)**k * torch.sin(phis)**(ni-k)
                bi[k, :] = bik / torch.linalg.norm(bik) #Normalize for numerical stability
            self.natural_basis.append(bi)
            self.on_basis.append(get_ON(bi))
            # print(f"this passed {ni}, dim:{bi.shape[0]}")
    
    def __repr__(self) -> str:
        return f"""SinoMoments(
                n_moments={self.n_moments}
                geometry={self.geometry}
        )"""

    def get_moment(self, X: torch.Tensor, ni: int):
        """
            Batch_size x sinoshape -> Batch_size x phi_size (moments) 
        """
        N, phi_size, t_size = X.shape
        assert ni < self.n_moments
        
        t = torch.from_numpy(self.geometry.detector_partition.grid.__array__()[:, 0]).to(DEVICE, dtype=X.dtype)
        k = t**ni       

        return torch.sum(X*k, dim=-1)
    
    def project_moment(self, M: torch.Tensor, ni: int):
        N, phi_size = M.shape
        assert ni < self.n_moments

        oni = self.on_basis[ni]
        projs_c = torch.einsum("ik,jk->ij", M, oni) #projection values onto basis

        return torch.einsum("ij,jk", projs_c, oni) #linear combination of basis
    

    def reduce_space(self, basis: torch.Tensor, ni: int, maintain_on = True):
        """
            From the space spanned by basis (assumed to be in sinogram space) find the subspace who satisfies the ni:th moment constraint.
            If basis is orthonormal and maintain_on is True the basis of the subspace will also be orthonormal.

            Args:
                basis (tensor n x phi_size x t_size): basis to reduce from
                ni (int): moment constraint to reduce with
        """
        n, phi_size, t_size = basis.shape
        assert phi_size == self.geometry.phi_size and t_size == self.geometry.t_size
        assert torch.linalg.matrix_rank(basis.reshape(n, phi_size*t_size)) == n

        A = self.get_moment(basis, ni).T #linear transformation from coordinates in basis to Moment
    
        oni_columnwise = self.on_basis[ni].T
        coords = get_null_space(torch.concat([A, oni_columnwise], dim=1))[:, :n]
        
        coords = get_ON(coords, degenerate=True)

        return torch.einsum("ik,kxy->ixy", coords, basis)



def get_null_space(A: torch.Tensor)->torch.Tensor:
    "Returns a tensor in which the rows are a basis for the nullspace of A."
    assert len(A.shape) == 2

    u, S, vh = torch.linalg.svd(A)
    nonzeros = torch.where(S > ZERO_THRESHOLD)[0].shape[0]
    return vh[nonzeros:, :]

def get_ON(basis: torch.Tensor, B: torch.Tensor = None, degenerate = False):
    """
        Return an orthonormal basis for the subspace spanned by the rows of basis, assuming a standard inner product or using the bilinear form B to define it.

        parameters:
            - basis (n x d Tensor): list of basis tensors for the subspace to find an ON basis for
            - B (n x n Tensor): matrix defining the inner product. Entry i, j should be the inner product
                                between the i:th and j:th basis vectors listed in 'basis'. If omitted the stamdard
                                inner product is used.
            - degenerate (bool): if False the inner product must be positive definite.
                                If True (e.g if basis is linearly dependant) the basis an orthonormal basis for the subspace in which it is positive definite will be returned 
                                
    """
    n, d = basis.shape
    assert d >= n and torch.linalg.matrix_rank(basis) == n, "basis must be linearly independant"
    if B is None:
        B = torch.tensor([
            [torch.sum(basis[i]*basis[j]) for j in range(n)] for i in range(n)
        ], device=DEVICE)
    else:
        assert B.shape == (n,n) and torch.dist(B.mH,B) < ZERO_THRESHOLD, "B should be a hermitian positive definite matrix defining the inner product"

    # Q = torch.linalg.inv(torch.linalg.cholesky(B).mH)
    L, Q = torch.linalg.eigh(B) #A = Q @ diag(L) @ Q^* (adjoint)
    if degenerate:
        assert (L > -ZERO_THRESHOLD).all(), "B must be positive semidefinite, even if degenerate"
        num_zeros = torch.where(L < ZERO_THRESHOLD)[0].shape[0]
        L = L[num_zeros:]
        Q = Q[:, num_zeros:]
    else:
        assert (L > ZERO_THRESHOLD).all(), "B must be positive definite!"
        
    on = torch.einsum("ij,ik->jk", Q, basis) / torch.sqrt(L)[:, None]

    return on

def project(X: torch.Tensor, on: torch.Tensor):
    "Project batch of tensors. Tensors are expected to represent discretized functions in Rn."
    N = X.shape[0]
    Xflat = X.reshape(N,-1)
    onflat = on.reshape(on.shape[0], -1)
    projs_c = torch.einsum("ik,jk->ij", Xflat, onflat) #projection values onto basis

    return torch.einsum("ij,jk", projs_c, onflat).reshape(*X.shape) #linear combination of basis


def trigo_sino_basis(geometry: Geometry, n_phi_freqs: int = 20, n_t_freqs: int = 20):
    assert geometry.ar == 1.0
    Nt, Np = geometry.t_size, geometry.phi_size
    art_t = torch.linspace(0.0, 2*torch.pi, steps=Nt, device=DEVICE)
    art_phi = torch.linspace(0.0, 2*torch.pi, steps=Np, device=DEVICE)

    basis = torch.zeros(4*n_phi_freqs*n_t_freqs, Np, Nt)

    for fp in range(n_phi_freqs):
        for ft in range(n_t_freqs):
            basis[4*(fp*n_t_freqs + ft), :, :] = torch.cos(fp*art_phi)[:, None] * torch.cos(ft*art_t)
            if fp > 0 and ft > 0:
                basis[4*(fp*n_t_freqs + ft)+1, :, :] = torch.sin(fp*art_phi)[:, None] * torch.cos(ft*art_t) 
                basis[4*(fp*n_t_freqs + ft)+2, :, :] = torch.cos(fp*art_phi)[:, None] * torch.sin(ft*art_t)
                basis[4*(fp*n_t_freqs + ft)+3, :, :] = torch.sin(fp*art_phi)[:, None] * torch.sin(ft*art_t)
    #remove zero rows from sin of zero freq
    norms = torch.einsum("ikl,ikl->i", basis, basis)
    basis = basis[norms > ZERO_THRESHOLD, :, :] / torch.sqrt(norms[norms > ZERO_THRESHOLD, None, None])
    return basis

def onehot_sino_basis(geometry: Geometry):
    Nt, Np = geometry.t_size, geometry.phi_size
    basis = torch.zeros(Np*Nt, Np, Nt)
    for i in range(Np):
        for j in range(Nt):
            basis[i*Nt + j, i, j] = 1

    return basis

def hat_basis(geometry: Geometry, n_phi_hats: int = 20, n_t_hats: int = 20):
    "get ON basis of flat surfaces"
    assert geometry.ar == 1.0
    Nt, Np = geometry.t_size, geometry.phi_size

    tstep, pstep = Nt // n_t_hats, Np // n_phi_hats
    basis = torch.zeros(n_t_hats * n_phi_hats, Np, Nt, device=DEVICE)
    for i in range(n_phi_hats):
        lp = i*pstep
        up = (i+1)*pstep if i < n_phi_hats-1 else Np
        for j in range(n_t_hats):
            lt = j*tstep
            ut = (j+1)*tstep if j < n_t_hats-1 else Nt
            basis[i*n_t_hats + j, lp:up, lt:ut] = 1
    
    norms_sq = torch.einsum("ikl,ikl->i", basis, basis)
    basis = basis / torch.sqrt(norms_sq[:, None, None])

    return basis

    

def intersection(basis1: torch.Tensor, basis2: torch.Tensor):
    n1, n2 = basis1.shape[0], basis2.shape[0]
    vecshape = basis1.shape[1:]
    return get_null_space(torch.concat([basis1.reshape(n1, -1), basis2.reshape(n2,-1)]).T).reshape(-1,*vecshape)

if __name__ == '__main__':
    import odl.contrib.torch as odl_torch
    import matplotlib.pyplot as plt
    import torch.nn as F

    g = Geometry(0.5, 100, 100)
    ext_g = extend_geometry(g)
    full_ray_layer = odl_torch.OperatorModule(ext_g.ray)
    N_moments = 5
    sm = SinoMoments(ext_g, N_moments)

    train_sinos, train_y, test_sinos, test_y = setup(g, num_to_generate=0, use_realistic=True, data_path="data/kits_phantoms_256.pt")
    full_test_sinos = full_ray_layer(test_y)
    mse = lambda diff : torch.mean(diff**2)

    # full_basis = trigo_sino_basis(ext_g, 6, 6)
    # full_basis = onehot_sino_basis(ext_g)
    full_basis = hat_basis(ext_g, 10, 10)
    basis = full_basis
    print("Basis constructed of dimension ", full_basis.shape[0])

    for ni in range(N_moments):
        basis = sm.reduce_space(basis, ni)
        print(f"reduced {ni}")
    
    print("final dimension is", basis.shape[0])

    n = basis.shape[0]
    trunced = basis[:, :g.phi_size, :].reshape(n, -1)
    # trunced = trunced / torch.sum(trunced**2, dim=-1)[:, None]
    proj_B = trunced @ trunced.mH
    proj_basis = get_ON(basis.reshape(n, -1), B = proj_B, degenerate=True).reshape(-1, ext_g.phi_size, ext_g.t_size)
    print("Projectin basis dimension is ", proj_basis.shape[0])

    padding = torch.zeros(test_sinos.shape[0], ext_g.phi_size-g.phi_size, g.t_size)
    padded_sinos = torch.concat([test_sinos, padding], dim=1)

    proj_sinos = project(full_test_sinos, basis)
    exp_sinos = project(padded_sinos, basis)

    for ni in range(N_moments):
        moms = sm.get_moment(full_test_sinos, ni)
        proj_moms = sm.project_moment(moms, ni)
        moms_padded = sm.get_moment(padded_sinos, ni)
        proj_moms_padded = sm.project_moment(moms_padded, ni)
        moms_exp = sm.get_moment(exp_sinos, ni)
        proj_moms_exp = sm.project_moment(moms_exp, ni)

        print(
            f"difference between true and projected moment of degree {ni} for full sinos: ", mse(moms-proj_moms), " zero padded sinos: ", mse(moms_padded-proj_moms_padded), " and for exp sinos: ", mse(moms_exp-proj_moms_exp)
            )


    print("="*40)
    print(f"mse between true and projected sinograms {mse(full_test_sinos-proj_sinos)}")
    print(f"mse between true and expanded sinograms {mse(full_test_sinos-exp_sinos)}")
    print(f"mse between true and zero padded sinograms {mse(full_test_sinos-padded_sinos)}")



    plt.subplot(131)
    plt.imshow(full_test_sinos[0])

    plt.subplot(132)
    plt.imshow(proj_sinos[0])

    plt.subplot(133)
    plt.imshow(exp_sinos[0])

    plt.show()