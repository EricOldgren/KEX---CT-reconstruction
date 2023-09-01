import torch
from abc import ABC, abstractmethod
from typing import Iterable, Tuple, Type

from utils.tools import no_bdry_linspace

class PolynomialBase(ABC):
    "Family of polynomials orthogonal on the interval [-a,a] with weight function W"
    a: float
    key: int
    "family id. Used when saving to recover Polynomial family."

    def __init__(self, a: float) -> None:
        self.a = a
    @abstractmethod
    def iterate_polynomials(self, N: int, x: torch.Tensor)->Iterable[Tuple[torch.Tensor, float]]:
        "iterate through the first N polynomials of this famliy, yield a polynomial sampled at the given xs and the square of its L2 norm"
    
    @abstractmethod
    def w(self, x: torch.Tensor)->torch.Tensor:
        "weight function for inner product sampled at the given xs"

class Legendre(PolynomialBase):
    key = 0

    def iterate_polynomials(self, N: int, x: torch.Tensor):
        "iterate through the N first legendre polynomials"
        vs = x / self.a
        pa = vs*0 +1
        pb = vs*1
        k = 0
        while k < N:
            #loop invariant pa is P_k, pb is P_{k+1}
            yield pa, 2/(2*k+1)*self.a
            k += 1
            pa, pb = pb, ((2*k+1)*vs*pb - k*pa) / (k+1) #based on recursive formula from Wikipedia
    
    def w(self, x: torch.Tensor):
        return x*0+1

class Chebyshev(PolynomialBase):
    key = 1

    def iterate_polynomials(self, N: int, x: torch.Tensor):
        k = 0
        vs = x / self.a
        torch.acos(vs, out=vs)
        res = vs*0
        W = self.w(x)

        while k < N:
            torch.mul(vs, k+1, out=res)
            torch.sin(res, out=res)
            res /= W
            yield res, torch.pi/2*self.a #torch.sin((k+1)*torch.acos(normalized_x)) / torch.sqrt(1-normalized_x**2)
            k += 1
    
    def w(self, x: torch.Tensor):
        res = x / self.a
        res *= res
        res *= -1
        res += 1
        torch.sqrt(res, out=res)
        return res #sqrt(1-(x/a)^2)

ALL_AVAILABLE_POLYNOMIAL_FAMILIES: Tuple[Type[PolynomialBase]] = (Legendre, Chebyshev)
POLYNOMIAL_FAMILY_MAP = {
    f.key: f for f in ALL_AVAILABLE_POLYNOMIAL_FAMILIES
}

def linear_upsample_inside(X: torch.Tensor, factor = 10):
    """
        Consider X as a batch of 1d functions and generate the tensor corresponding to a sample with factor times as many samples as X along the last dimension. Intermediate values are obtained via linear interpolation.
        Exact size of upsampled tensor along last dimension is (N-1)*factor+1 since it is impossible to interpolate beyond the ends 
    
        args:
            X (Tensor) of any shape
            factor (int)
    """
    pre_shape, N = X.shape[:-1], X.shape[-1]
    dX = X[...,1:] - X[...,:-1]
    dX /= factor
    vals = X+0 #copy
    res = torch.zeros(pre_shape + ((N-1)*factor+1,), dtype=X.dtype, device=X.device)

    for i in range(factor):
        res[..., i::factor] = vals
        if i == 0:
            vals = vals[...,:-1] #last element has no trailing interpolated values 
        vals += dX

    return res

def linear_upsample_no_bdry(X: torch.Tensor, factor = 11):
    "Upsampling when discretization does not have points on boundary, factor must be odd. result will have shape (*, last_dim*factor)"

    assert factor % 2 == 1, "factor must be odd for no_boundary linear upsampling"
    pre_shape, N = X.shape[:-1], X.shape[-1]
    dX = X*0
    dX[...,:-1] += X[...,1:]
    dX[...,:-1] -= X[...,:-1]
    dX[...,-1] = dX[...,-2] #resuse slope beyond last sampled point
    # dX = X[...,1:]-X[...,:-1]
    dX /= factor
    res = torch.zeros(pre_shape + (N*factor,), dtype=X.dtype, device=X.device)

    vals = X+0 #Copy
    for i in range(factor):
        if i == factor//2+1:
            vals = vals[...,:-1]
            dX = dX[...,:-1]
        res[...,factor//2+i::factor] = vals
        vals += dX

    pre_vals = X[...,0] - dX[...,0]
    for i in range(factor//2):
        res[...,factor//2-i-1] = pre_vals
        pre_vals -= dX[...,0]

    return res

def down_sample_inside(X: torch.Tensor, factor = 10):
    """
        inverse of linear_upsample
        pick out every factor:th element along the last dimension
    """
    return X[..., ::factor]
def down_sample_no_bdry(X: torch.Tensor, factor = 11):
    "inverse of `linear_upsample_no_bdry`"
    assert factor % 2 == 1, "factor must be odd"

    return X[...,factor//2::factor]

def get_prim(f: torch.Tensor):
        n, = f.shape
        res = f[None].repeat(n,1)
        ivals, jvals = torch.arange(0,n)[:,None].repeat(1,n), torch.arange(0,n)[None].repeat(n,1)
        res[jvals>ivals] = 0
        return torch.sum(res, dim=-1)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    Nx = 50
    max_deg = 100
    factor = 31
    func = lambda xs : (1-xs**2)**10 #sum(torch.cos(k*xs*torch.pi) for k in range(20))

    x_full_bdry = torch.linspace(-1,1, Nx*factor)
    f_full_bdry = func(x_full_bdry)
    x_bdry = torch.linspace(-1.0,1.0, Nx)
    f_bdry = func(x_bdry)
    x_upsampled_bdry = linear_upsample_inside(x_bdry, factor)
    f_upsampled_bdry = linear_upsample_inside(f_bdry, factor)

    x_full_no_bdry = no_bdry_linspace(-1,1,Nx*factor)
    f_full_no_bdry = func(x_full_no_bdry)
    x_no_bdry = no_bdry_linspace(-1,1, Nx)
    f_no_bdry = func(x_no_bdry)
    x_upsampled_no_bdry = linear_upsample_no_bdry(x_no_bdry, factor)
    f_upsampled_no_bdry = linear_upsample_no_bdry(f_no_bdry, factor)
             

    def project_with_pols(x: torch.Tensor, f: torch.Tensor, N:int, title:str):
        dx = torch.mean(x[1:]-x[:-1])
        legendres = Legendre(1.0)

        L = torch.stack([pn/np.sqrt(norm_sq) for pn, norm_sq in legendres.iterate_polynomials(N, x)])
        B = torch.einsum("nx,mx->nm", L*dx, L)
        print("="*40)
        print("Eval data", title)
        print("orth mse:", torch.mean((B-torch.eye(N))**2))
        print(B[:5, :5])

        #Projecting f
        g = f *0
        for n, (pn, norm_sq) in enumerate(legendres.iterate_polynomials(N, x)):
            #g += torch.sum(f*pn*dx) / torch.sum(pn**2*dx) * pn
            g += torch.sum(f*pn*dx) / norm_sq * pn

        print("="*40)
        return g
    
    g_full = project_with_pols(x_full_bdry, f_full_bdry, max_deg, "full")
    g = project_with_pols(x_bdry, f_bdry, max_deg, "sparse")
    g_upsampled = project_with_pols(x_upsampled_bdry, f_upsampled_bdry, max_deg, "upsampled")
    g_up_down = down_sample_inside(g_upsampled, factor)

    mse_fn = lambda diff : torch.mean(diff**2)

    print("Mses with boundary")
    print("full", mse_fn(g_full-f_full_bdry).item())
    print("sparse", mse_fn(g-f_bdry).item())
    print("upsampled", mse_fn(g_upsampled-f_upsampled_bdry).item())
    print("up down", mse_fn(g_up_down-f_bdry))

    fig, _ = plt.subplots(1,3)
    plt.subplot(131)
    plt.title("full data")
    plt.plot(x_full_bdry, f_full_bdry, label="gt")
    plt.plot(x_full_bdry, g_full, label="projection")
    plt.legend()
    
    plt.subplot(132)
    plt.title("sparse data")
    plt.plot(x_bdry, f_bdry, label="gt")
    plt.plot(x_bdry, g, label="projection")
    plt.plot(x_bdry, g_up_down, label="up down")
    plt.legend()
    
    plt.subplot(133)
    plt.title("upsampled data")
    plt.plot(x_upsampled_bdry, f_upsampled_bdry, label="gt")
    plt.plot(x_upsampled_bdry, g_upsampled, label="projection")
    plt.legend()
    fig.show()

    ##
    ##No boundary stuff
    ##
    g_full_no_bdry = project_with_pols(x_full_no_bdry, f_full_no_bdry, max_deg, "full no boundary")
    g_no_bdry = project_with_pols(x_no_bdry, f_no_bdry, max_deg, "sparse no boundary")
    g_upsampled_no_bdry = project_with_pols(x_upsampled_no_bdry, f_upsampled_no_bdry, max_deg, "upsampled no boundary")
    g_up_down_no_bdry = down_sample_no_bdry(g_upsampled_no_bdry, factor)
    
    print("Mses without boundary")
    print("full", mse_fn(g_full_no_bdry-f_full_no_bdry).item())
    print("sparse", mse_fn(g_no_bdry-f_no_bdry).item())
    print("upsampled", mse_fn(g_upsampled_no_bdry-f_upsampled_no_bdry).item())
    print("up down", mse_fn(g_up_down_no_bdry-f_no_bdry))

    fig, _ = plt.subplots(1,3)
    plt.subplot(131)
    plt.title("full data no bdry")
    plt.plot(x_full_no_bdry, f_full_no_bdry, label="gt")
    plt.plot(x_full_no_bdry, g_full_no_bdry, label="projection")
    plt.legend()
    
    plt.subplot(132)
    plt.title("sparse data no bdry")
    plt.plot(x_no_bdry, f_no_bdry, label="gt")
    plt.plot(x_no_bdry, g_no_bdry, label="projection")
    plt.plot(x_no_bdry, g_up_down_no_bdry, label="up down")
    plt.legend()
    
    plt.subplot(133)
    plt.title("upsampled data no bdry")
    plt.plot(x_upsampled_no_bdry, f_upsampled_no_bdry, label="gt")
    plt.plot(x_upsampled_no_bdry, g_upsampled_no_bdry, label="projection")
    plt.legend()
    fig.show()

    plt.show()