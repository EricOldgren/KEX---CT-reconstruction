import torch

from utils.geometry import Geometry, setup, DEVICE

ZERO_THRESHOLD = 1e-3

class SinoMoments:

    def __init__(self, geometry: Geometry, n_moments = 12) -> None:
        self.geometry = geometry
        assert geometry.ar == 1.0, "assumng full geometry"
        self.n_moments = n_moments

        phis = torch.linspace(0.0, torch.pi, steps=geometry.phi_size, device=DEVICE)
        self.natural_basis = []
        "list of rank 2 tensors, each a basis spanning the subspace in which the n:th moment should lie in. Basis listed rowwise"
        self.on_basis = []
        "list of ON bases, each basis rowwise"
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
            If basis is orthonormal and the maintain_on is True the basis of the subspace will also be orthonormal.

            Args:
                basis (tensor n x phi_size x t_size): basis to redice from
                ni (int): moment constraint to reduce with
        """
        n, phi_size, t_size = basis.shape
        assert phi_size == self.geometry.phi_size and t_size == self.geometry.t_size
        assert torch.linalg.matrix_rank(basis.reshape(n, phi_size*t_size)) == n

        A = self.get_moment(basis, ni).T #linear transformation from coordinates in basis to Moment
        Ap = torch.linalg.pinv(A)

        oni_columnwise = self.on_basis[ni].T
        pinved = (Ap@oni_columnwise).T
        if maintain_on: pinved = get_ON(pinved)
        coords = torch.concatenate([pinved, get_null_space(A)], dim=0)

        return torch.einsum("ik,kxy->ixy", coords, basis)



def get_null_space(A: torch.Tensor)->torch.Tensor:
    "Returns a tensor in which the rows are a basis for the nullspace of A."
    assert len(A.shape) == 2

    u, S, vh = torch.linalg.svd(A)
    nonzeros = torch.where(S > ZERO_THRESHOLD)[0].shape[0]
    return vh[nonzeros:, :]

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
        ], device=DEVICE)
    else:
        assert B.shape == (n,n) and B.mH == B, "B should be a hermitian positive definite matrix defining the inner product"

    # Q = torch.linalg.inv(torch.linalg.cholesky(B).mH)
    L, Q = torch.linalg.eigh(B) #A = Q @ diag(L) @ Q^* (adjoint)
    assert (L > 0).all(), "B must be positive definite!"
    
    on = torch.einsum("ij,ik->jk", Q, basis) / torch.sqrt(L)[:, None]

    return on

def project(X: torch.Tensor, on: torch.Tensor):
    "Project batch of tensors. Tensors are expected to represent discretized functions in Rn."
    N = X.shape[0]
    Xflat = X.reshape(N,-1)
    onflat = on.reshape(-1, Xflat.shape[1])
    projs_c = torch.einsum("ik,jk->ij", Xflat, onflat) #projection values onto basis

    return torch.einsum("ij,jk", projs_c, onflat).reshape(*X.shape) #linear combination of basis


def trigo_sino_basis(geometry: Geometry, n_t_freqs: int = 20, n_phi_freqs: int = 20):
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

if __name__ == '__main__':

    g = Geometry(1.0, 200, 100)
    N_moments = 15
    sm = SinoMoments(g, N_moments)

    train_sinos, train_y, test_sinos, test_y = setup(g, num_to_generate=50, use_realistic=False, data_path="data/kits_phantoms_256.pt")
    mse = lambda diff : torch.mean(diff**2)

    for ni in range(N_moments):
        moms = sm.get_moment(test_sinos, ni)
        proj_moms = sm.project_moment(moms, ni)

        print(f"difference between true and projected moment of degree {ni}: ", mse(moms-proj_moms))
    print("="*40)

    Nt, Np = g.t_size, g.phi_size
    basis = trigo_sino_basis(g, 10, 10)
    print("Basis constructed")

    for ni in range(N_moments):
        basis = sm.reduce_space(basis, ni)
        print(f"reduced {ni}")

    proj_sinos = project(test_sinos, basis)

    print(f"mse between true and projected sinograms {mse(test_sinos-proj_sinos)}")

    