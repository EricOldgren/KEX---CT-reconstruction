import torch

from utils.geometry import Geometry, setup, DEVICE

class SinoMoments:

    def __init__(self, geometry: Geometry, n_moments = 12) -> None:
        self.geometry = geometry
        assert geometry.ar == 1.0, "assumng full geometry"
        self.n_moments = n_moments

        phis = torch.linspace(0.0, torch.pi, steps=geometry.phi_size, device=DEVICE)
        self.natural_basis = []
        "list of rank 2 tensors, each a basis spanning the subspace in which the n:th moment should lie in"
        self.on_basis = []
        for ni in range(n_moments):
            #construct basis for homogenous polynomials of degree n in sin and cos
            bi = torch.zeros(ni+1, geometry.phi_size, device=DEVICE)
            for k in range(0, ni+1):
                bik = torch.cos(phis)**k * torch.sin(phis)**(ni-k)
                bi[k, :] = bik / torch.linalg.norm(bik) #Normalize for numerical stability
            self.natural_basis.append(bi)
            self.on_basis.append(get_ON(bi))
            # print(f"this passed {ni}, dim:{bi.shape[0]}")
    
    def get_moment(self, X: torch.Tensor, ni: int):
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

def get_ON(basis: torch.Tensor, B: torch.Tensor = None):
    """
        Return an orthonormal basis for the subspace spanned by the rows of basis, assuming a standard inner product or using the bilinear form B to define it.

        parameters:
            - basis (n x d Tensor): list of basis tensors for the subspace to find an ON basis for
            - B (n x n Tensor): matrix defining the inner product. Entry i, j should be the inner product between the i:th and j:th basis vectors listed in 'basis' 
    """
    n, d = basis.shape
    assert d >= n and torch.linalg.matrix_rank(basis) == n, "basis must be linearly independant"
    if B is None:
        B = torch.tensor([
            [torch.sum(basis[i]*basis[j]) for j in range(n)] for i in range(n)
        ])
    else:
        assert B.shape == (n,n) and B.T.conj() == B, "B should be a hermitian positive definite matrix defining the inner product"
    
    L, Q = torch.linalg.eigh(B) #A = Q @ diag(L) @ Q^* (adjoint)
    assert (L > 0).all(), "B must be positive definite"
    
    on = torch.einsum("ij,ik->jk", Q, basis) / torch.sqrt(L)[:, None]

    return on


if __name__ == '__main__':

    g = Geometry(1.0, 200, 100)
    sm = SinoMoments(g, 15)

    train_sinos, train_y, test_sinos, test_y = setup(g, num_to_generate=0, use_realistic=True, data_path="data/kits_phantoms_256.pt")
    mse = lambda diff : torch.mean(diff**2)

    for i in range(15):
        moms = sm.get_moment(test_sinos, i)
        proj_moms = sm.project_moment(moms, i)

        print(f"difference moment {i}: ", mse(moms-proj_moms))