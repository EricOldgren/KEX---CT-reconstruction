import torch
from abc import ABC, abstractmethod
from typing import Iterable, Tuple

class PolynomialBase(ABC):
    "Family of polynomials orthogonal on the interval [-a,a] with weight function W"
    a: float

    def __init__(self, a: float) -> None:
        self.a = a
    @abstractmethod
    def iterate_polynomials(self, N: int, x: torch.Tensor)->Iterable[Tuple[torch.Tensor, float]]:
        "iterate through the first N polynomials of this famliy, yield a polynomial sampled at the given xs and the square of its L2 norm"
    
    @abstractmethod
    def w(self, x: torch.Tensor)->torch.Tensor:
        "weight function for inner product sampled at the given xs"

class Legendre(PolynomialBase):

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

def linear_upsample_no_bdry(X: torch.Tensor, factor = 10):
    return

def down_sample(X: torch.Tensor, factor = 10):
    """
        inverse of linear_upsample
        pick out every factor:th element along the last dimension
    """
    return X[..., ::factor]

def get_prim(f: torch.Tensor):
        n, = f.shape
        res = f[None].repeat(n,1)
        ivals, jvals = torch.arange(0,n)[:,None].repeat(1,n), torch.arange(0,n)[None].repeat(n,1)
        res[jvals>ivals] = 0
        return torch.sum(res, dim=-1)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    Nx = 500
    max_deg = 100
    factor = 5
    func = lambda xs : sum(torch.sin(k*xs*torch.pi) for k in range(10))
    x_full = torch.linspace(-1,1, Nx*factor)
    dx_full = torch.mean(x_full[1:] - x_full[:-1])
    f_full = func(x_full)

    x = torch.linspace(-1.0,1.0, Nx)
    dx = torch.mean(x[1:]-x[:-1])
    f = func(x)
    x_upsampled = linear_upsample_inside(x, factor)
    f_upsampled = linear_upsample_inside(f, factor)
    # dx = 2.0 / Nx
    # x = -1.0 + dx/2 + torch.arange(0,Nx)*dx
    # dx = torch.mean(x[1:] - x[:-1])
    # dx /= factor
    # f = linear_upsample_inside(f, factor)
    # x = linear_upsample_inside(x, factor)
    # g = f*0
    
         

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
    
    g_full = project_with_pols(x_full, f_full, max_deg, "full")
    g = project_with_pols(x, f, max_deg, "sparse")
    g_upsampled = project_with_pols(x_upsampled, f_upsampled, max_deg, "upsampled")
    g_up_down = down_sample(g_upsampled, factor)

    print("Mses")
    mse_fn = lambda diff : torch.mean(diff**2)
    print("full", mse_fn(g_full-f_full).item())
    print("sparse", mse_fn(g-f).item())
    print("upsampled", mse_fn(g_upsampled-f_upsampled).item())
    print("up down", mse_fn(g_up_down-f))

    plt.subplot(131)
    plt.title("full data")
    plt.plot(x_full, f_full, label="gt")
    plt.plot(x_full, g_full, label="projection")
    plt.legend()
    
    plt.subplot(132)
    plt.title("sparse data")
    plt.plot(x, f, label="gt")
    plt.plot(x, g, label="projection")
    plt.plot(x, g_up_down, label="up down")
    plt.legend()
    
    plt.subplot(133)
    plt.title("upsampled data")
    plt.plot(x_upsampled, f_upsampled, label="gt")
    plt.plot(x_upsampled, g_upsampled, label="projection")
    plt.legend()

    plt.show()
    
