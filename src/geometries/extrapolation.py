import torch
from geometries.data import get_htc2022_train_phantoms
from utils.polynomials import Chebyshev, Legendre
from utils.tools import MSE, DEVICE, CDTYPE, DTYPE
from geometries.geometry_base import FBPGeometryBase
from geometries.fanbeam_geometry import get_moment_mask
from tqdm import tqdm



def extrapolate_fixpoint(la_sinos: torch.Tensor, known_region: torch.Tensor, geometry: FBPGeometryBase, M: int, K: int, n_iters = 1000, PolynomialFamily = Legendre):
    """Extrapolate sinos based on HLCC using fix point iteration

    Args:
        la_sinos (torch.Tensor): _description_
        known_angles (torch.Tensor): _description_
        geometry (FBPGeometryBase): _description_
        M (int): _description_
        K (int): _description_
        n_iters (int, optional): _description_. Defaults to 1000.
        PolynomialFamily (_type_, optional): _description_. Defaults to Legendre.

    Returns:
        estimated full angle sinograms
    """
    batch_size = la_sinos.shape[0]
    embedding = torch.zeros((batch_size, M, K), device=DEVICE, dtype=CDTYPE)
    mask = get_moment_mask(embedding)
    cn = embedding[:, mask] + 0

    print("Extrapolating sinos via HLCC")
    for it in tqdm(range(n_iters)):
        embedding[:, mask] = cn
        exp = geometry.synthesise_series(embedding, PolynomialFamily)
        exp[:, ~known_region] = 0 #Bcn

        cn += geometry.series_expand(la_sinos-exp, PolynomialFamily, M, K)[:, mask]

    embedding[:, mask] = cn
    return geometry.synthesise_series(embedding, PolynomialFamily)

def extrapolate_direct_solve(la_sinos: torch.Tensor, known_region: torch.Tensor, geometry: FBPGeometryBase, M: int, K: int, u: torch.Tensor = None, l2_reg = 0.2, PolynomialFamily = Legendre):
    """Extrapolate sinos based on HLCC by directly solving the normal equations for expanding sinos in the known region.
    The matrix used to solve this is factorized via Cholesky decomposition - this can be precomputed and given as an input argument.

    Args:
        la_sinos (torch.Tensor): _description_
        known_angles (torch.Tensor): _description_
        geometry (FBPGeometryBase): _description_
        M (int): _description_
        K (int): _description_
        u (torch.Tensor, optional): precomputed Cholesky decomposition of the matrix appearing in the normal equations. Defaults to None (i.e compute it).
        l2_reg (float, optional): _description_. Defaults to 0.2.
        PolynomialFamily (_type_, optional): _description_. Defaults to Legendre.

    Returns:
        extimated_sinos, u
    """


    batch_size = la_sinos.shape[0]
    embedding = torch.zeros((batch_size, M, K), device=DEVICE, dtype=CDTYPE)
    mask = get_moment_mask(embedding)
    n_coeffs = mask.sum()

    if u is None:
        B = torch.zeros((n_coeffs, n_coeffs), device=DEVICE, dtype=CDTYPE)
        mks = torch.cartesian_prod(torch.arange(0,M), torch.arange(0,K)).to(DEVICE).reshape(M, K, 2)[mask]
        print("constructing matrix")
        for i, (m, k) in tqdm(enumerate(mks)):
            inp = torch.zeros((1,M,K)).to(DEVICE, dtype=CDTYPE)
            inp[:, m, k] = 1
            x = geometry.synthesise_series(inp, Legendre)
            x[:, ~known_region] *= 0
            B[:, i] = geometry.series_expand(x, Legendre, M, K)[0, mask]
        print("Matrix constructed")
        u = torch.linalg.cholesky(B + l2_reg* torch.eye(n_coeffs, device=DEVICE))
    else:
        assert u.shape == (n_coeffs, n_coeffs)

    cs = torch.cholesky_solve(geometry.series_expand(la_sinos, Legendre, M, K)[:, mask].reshape(batch_size, n_coeffs, 1), u).reshape(batch_size, n_coeffs)
    embedding[:, mask] = cs
    return geometry.synthesise_series(embedding, PolynomialFamily), u

def extrapolate_cgm(la_sinos: torch.Tensor, known_region: torch.Tensor, geometry: FBPGeometryBase, M: int, K: int, PolynomialFamily = Legendre, tol = 0.1):
    """Extrapolate sinos based on HLCC using conjugate gradiend method to solve the normal equations.

    Args:
        la_sinos (torch.Tensor): _description_
        ar (float): _description_
        geometry (FBPGeometryBase): _description_
        M (int): _description_
        K (int): _description_
        PolynomialFamily (_type_, optional): _description_. Defaults to Legendre.
        tol (float, optional): _description_. Defaults to 0.1.

    Returns:
        estimated_sinos
    """
    
    batch_size = la_sinos.shape[0]
    embedding = torch.zeros((batch_size, M, K), device=DEVICE, dtype=CDTYPE)
    mask = get_moment_mask(embedding)

    ck =  embedding[:, mask] + 0
    rk = geometry.series_expand(la_sinos, PolynomialFamily, M, K)[:, mask]
    pk = rk + 0

    k = 0
    while (torch.linalg.norm(rk, dim=-1) > tol).any():

        #Calculat Apk (=w)
        embedding[:, mask] = pk
        exp = geometry.synthesise_series(embedding, PolynomialFamily)
        exp[:, ~known_region] = 0
        w = geometry.series_expand(exp, PolynomialFamily, M, K)[:, mask] #Apk -  shape batch_size x n

        ak = (rk*rk.conj()).sum(dim=-1, keepdim=True) / (pk.conj()*w).sum(dim=-1, keepdim=True) # shape batch_size x 1
        cnext = ck + ak*pk
        #Calculate r
        embedding[:, mask] = cnext
        exp = geometry.synthesise_series(embedding, PolynomialFamily)
        exp[:, ~known_region] = 0
        rnext = geometry.series_expand(la_sinos-exp, PolynomialFamily, M, K)[:, mask]
        # rnext = rk - ak*w

        bk = (rnext.conj()*rnext).sum(dim=-1, keepdim=True) / (rk*rk.conj()).sum(dim=-1, keepdim=True) #shape: bacth_size x 1
        pnext = rnext + bk*pk

        ck = cnext
        rk = rnext
        pk = pnext
        k += 1

        ##Debug INFO
        print("k:", k, "res mse:", torch.linalg.norm(rk, dim=-1).max())#, "absolute mse:", torch.linalg.norm(err, dim=-1))
        ###
        

    embedding[:, mask] = ck
    return geometry.synthesise_series(embedding, PolynomialFamily)


def precompute_system_matrix(known_region: torch.Tensor, geometry: FBPGeometryBase, M: int, K: int, PolynomialFamily = Chebyshev):
    mask = get_moment_mask(torch.zeros((1,M,K)).to(DEVICE, dtype=CDTYPE))
    n_coeffs = mask.sum()

    B = torch.zeros((n_coeffs, n_coeffs), device=DEVICE, dtype=CDTYPE)
    mks = torch.cartesian_prod(torch.arange(0,M), torch.arange(0,K)).to(DEVICE).reshape(M, K, 2)[mask]
    for i, (m, k) in tqdm(enumerate(mks), desc="constructing normal matrix for sinogram extrapolation"):
        inp = torch.zeros((1,M,K)).to(DEVICE, dtype=CDTYPE)
        inp[:, m, k] = 1
        x = geometry.synthesise_series(inp, PolynomialFamily)
        x[:, ~known_region] *= 0
        B[:, i] = geometry.series_expand(x, PolynomialFamily, M, K)[0, mask]
    
    return B


class SinoFilling(torch.nn.Module):
  def __init__(self, geometry: FBPGeometryBase, M: int = 100, K: int = 100, n_iters = 300, PolynomialFamily = Legendre):
    super().__init__()
    self.geometry = geometry
    self.M = M
    self.K = K
    self.n_iters = n_iters
    self.PolynomialFamily = PolynomialFamily
    self.mask = get_moment_mask(torch.zeros((1,M, K), device=DEVICE))

  def forward(self, la_sinos: torch.Tensor, known_region: torch.Tensor):
    return extrapolate_fixpoint(la_sinos, known_region, self.geometry, self.M, self.K, self.n_iters, self.PolynomialFamily)
  

class RidgeSinoFiller(torch.nn.Module):

    def __init__(self, geometry: FBPGeometryBase, knwon_region: torch.Tensor, M: int = 50, K: int = 50, PolynomialFamily = Chebyshev) -> None:
        super().__init__()

        self.geometry = geometry
        self.known_region = knwon_region
        self.M, self.K = M, K
        self.PolynomialFamily = PolynomialFamily
        self.mask = get_moment_mask(torch.zeros((1,M,K)))
        self.n_coeffs = self.mask.sum()
        self.normal_matrix = torch.nn.Parameter(torch.zeros((self.n_coeffs, self.n_coeffs), device=DEVICE, dtype=CDTYPE), requires_grad=False)
        self._matrix_computed = torch.nn.Parameter(torch.tensor(False), requires_grad=False)

    def _compute_system_matrix(self, force_recompute = False):
        if ~self._matrix_computed or force_recompute:
            self.normal_matrix *= 0
            self.normal_matrix += precompute_system_matrix(self.known_region, self.geometry, self.M, self.K, self.PolynomialFamily)
            self._matrix_computed |= True
        return self.normal_matrix

    def forward(self, la_sinos: torch.Tensor, l2_reg = 0.01):
        B = self._compute_system_matrix()
        b = self.geometry.series_expand(la_sinos, self.PolynomialFamily, self.M, self.K)[:, self.mask].reshape(-1, self.n_coeffs, 1)

        c = torch.linalg.solve(B+torch.eye(self.n_coeffs, device=DEVICE, dtype=CDTYPE)*l2_reg, b)
        
        assert MSE(b, B@c) < 0.02, f"mse: {MSE(b, B@c)} if this breaks extrapolation is less effective than when tested :/ Comment this line if needed"

        embedding = torch.zeros((la_sinos.shape[0],self.M,self.K)).to(DEVICE, dtype=CDTYPE)
        embedding[:, self.mask] += c[...,0]
        return self.geometry.synthesise_series(embedding, self.PolynomialFamily)
    
class PrioredSinoFilling(torch.nn.Module):

    def __init__(self, geometry: FBPGeometryBase, knwon_region: torch.Tensor, M: int = 50, K: int = 50, PolynomialFamily = Chebyshev) -> None:
        super().__init__()

        self.geometry = geometry
        self.known_region = knwon_region
        self.M, self.K = M, K
        self.PolynomialFamily = PolynomialFamily
        self.mask = get_moment_mask(torch.zeros((1,M,K)))
        self.n_coeffs = self.mask.sum()
        self.normal_matrix = torch.nn.Parameter(torch.zeros((self.n_coeffs, self.n_coeffs), device=DEVICE, dtype=CDTYPE), requires_grad=False)
        self._matrix_computed = torch.nn.Parameter(torch.tensor(False), requires_grad=False)

        self.Z = torch.nn.Parameter(torch.zeros((self.n_coeffs, self.n_coeffs), device=DEVICE, dtype=CDTYPE), requires_grad=False)
        self.sigmas_sq = torch.nn.Parameter(torch.zeros(self.n_coeffs, device=DEVICE, dtype=DTYPE), requires_grad=False)
        self.mu = torch.nn.Parameter(torch.zeros(self.n_coeffs, device=DEVICE, dtype=CDTYPE), requires_grad=False)

    def _compute_system_matrix(self, force_recompute = False):
        if ~self._matrix_computed or force_recompute:
            self.normal_matrix *= 0
            self.normal_matrix += precompute_system_matrix(self.known_region, self.geometry, self.M, self.K, self.PolynomialFamily)
            self._matrix_computed |= True
        return self.normal_matrix
    
    def fit_prior(self, full_sinos: torch.Tensor, use_mu = True):
        "Estimates mean mu and principal components zi from the distribution of the coefficient expansion of given sinos. Any previous estimation of this is discarded."      
        self.Z *= 0
        self.sigmas_sq *= 0
        self.mu *= 0
        
        X = self.geometry.series_expand(full_sinos, self.PolynomialFamily, self.M, self.K)[:, self.mask] #shape: batch_size x n_coeffs
        batch_size, n_coeffs = X.shape
        if use_mu:
            self.mu += torch.mean(X, dim=0)
            X -= self.mu
        X /= batch_size
        sigmas_sq, zs = torch.linalg.eigh(X.mH@X) #eigh returns eigenvalues in ascending order!
        self.sigmas_sq += torch.flip(sigmas_sq, dims=(0,))
        self.Z += torch.flip(zs, dims=(1,))

        return self


    def forward(self, la_sinos: torch.Tensor, r: int, l2_reg = 0.01):
        B = self._compute_system_matrix()
        b = self.geometry.series_expand(la_sinos, self.PolynomialFamily, self.M, self.K)[:, self.mask].reshape(-1, self.n_coeffs, 1)
        if r > 0:
            w = self.sigmas_sq + 0
            if r is not None:
                w[r:] = w[r]
            W = self.Z @ torch.diag(w[0]/w).to(CDTYPE) @ self.Z.mH
        else:
            W = self.Z * 0

        c = torch.linalg.solve(B+l2_reg*W, b+l2_reg*W@self.mu)
        
        # assert MSE(b, B@c) < 0.02, f"mse: {MSE(b, B@c)} if this breaks extrapolation is less effective than when tested :/ Comment this line if needed"

        embedding = torch.zeros((la_sinos.shape[0],self.M,self.K)).to(DEVICE, dtype=CDTYPE)
        embedding[:, self.mask] += c[...,0]
        return self.geometry.synthesise_series(embedding, self.PolynomialFamily)