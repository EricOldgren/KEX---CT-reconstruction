from typing import Any
import torch


# @torch.jit.script - this made it slower ://
def _moment_projection(scaled_sinos: torch.Tensor, normalised_polynomials: torch.Tensor, W: torch.Tensor, phis2d: torch.Tensor):
    """
        Faster implementation of the critical step in projection of sinos onto subspace of valid sinograms.
    """
    
    res = scaled_sinos*0

    N_moments, _ = normalised_polynomials.shape
    trig_out = torch.zeros_like(phis2d)
    
    basises = torch.zeros_like(scaled_sinos)
    X = torch.zeros_like(scaled_sinos)
     
    for n in range(N_moments):    
        pn = normalised_polynomials[n]
        
        for k in range(n%2, n+1, 2):
            if k != 0:
                #Sin basis
                torch.mul(phis2d, k, out=trig_out)
                torch.sin(trig_out, out=trig_out)
                trig_out *= pn
                basises[...] = trig_out
                torch.mul(scaled_sinos, basises, out=X)
                projection_coeffs = torch.sum(X, dim=(-2,-1), keepdim=True)
                basises *= projection_coeffs / (torch.pi)
                basises *= W
                res += basises
            #Cos basis
            torch.mul(phis2d, k, out=trig_out)
            torch.cos(trig_out, out=trig_out)
            trig_out *= pn
            basises[...] = trig_out
            torch.mul(scaled_sinos, basises, out=X)
            projection_coeffs = torch.sum(X, dim=(-2,-1), keepdim=True)
            basises *= projection_coeffs / (torch.pi)
            if k == 0: basises /= 2
            basises *= W
            res += basises

    return res


class MomentProjectionFunction(torch.autograd.Function):
    """
        Moment Projection operator integrated with pytorch autograd. Only derivative w.r.t functions that are projected is implemented (not w.r.t ON-basis et.c.)
    """

    @staticmethod
    def forward(ctx: Any, scaled_sinos: torch.Tensor, normalised_polynomials: torch.Tensor, W: torch.Tensor, phis2d: torch.Tensor):
        assert not normalised_polynomials.requires_grad
        assert not W.requires_grad
        assert not phis2d.requires_grad

        ctx.normalised_polynomials= normalised_polynomials
        ctx.W = W
        ctx.phis2d = phis2d
        return _moment_projection(scaled_sinos, normalised_polynomials, W, phis2d)
    
    @staticmethod
    def backward(ctx: Any, grad_output) -> Any:
        return _moment_projection(grad_output, ctx.normalised_polynomials, ctx.W, ctx.phis2d), None, None, None

class MomentExpansionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, sinos: torch.Tensor, volume_scale: torch.Tensor, normalised_polynomials: torch.Tensor, W: torch.Tensor, phis2d: torch.Tensor, cdtype: torch.dtype, max_k: int = None):
        assert not volume_scale.requires_grad
        assert not normalised_polynomials.requires_grad
        assert not W.requires_grad
        assert not phis2d.requires_grad
        ctx.normalised_polynomials = normalised_polynomials
        ctx.W = W
        ctx.phis2d = phis2d
        
        return _sino_moment_expansion(sinos, volume_scale, normalised_polynomials, phis2d, cdtype, max_k)

    @staticmethod
    def backward(ctx: Any, grad_output):
        return _sino_moment_synthesis(grad_output, ctx.normalised_polynomials, ctx.W, ctx.phis2d), None, None, None, None, None
    
class MomentSynthesisFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, coefficients: torch.Tensor, normalised_polynomials: torch.Tensor, W: torch.Tensor, phis2d: torch.Tensor, volume_scale: torch.Tensor, cdtype: torch.dtype):
        assert not normalised_polynomials.requires_grad
        assert not W.requires_grad
        assert not phis2d.requires_grad
        assert not volume_scale.requires_grad

        _, _, k = coefficients.shape
        ctx.volume_scale = volume_scale
        ctx.normalised_polynomials = normalised_polynomials
        ctx.W = W
        ctx.phis2d = phis2d
        ctx.k = k
        ctx.cdtype = cdtype

        return _sino_moment_synthesis(coefficients, normalised_polynomials, W, phis2d)
    
    @staticmethod
    def backward(ctx: Any, grad_output):
        
        return _sino_moment_expansion(grad_output, ctx.volume_scale, ctx.normalised_polynomials, ctx.phis2d, ctx.cdtype, ctx.k), None, None, None, None, None

def _sino_moment_expansion(sinos: torch.Tensor, volume_scale: torch.Tensor, normalised_polynomials: torch.Tensor, phis2d: torch.Tensor, cdtype: torch.dtype, max_k: int = None):

    device, dtype = sinos.device, sinos.dtype

    normalised_polynomials = normalised_polynomials.to(cdtype) #einsum does not work when mixing real and complex operations
    scaled_sinos = sinos * volume_scale
    N, Np, Nu = scaled_sinos.shape
    N_moments, _ = normalised_polynomials.shape
    if max_k is None:
        max_k = N_moments
    res = torch.zeros((N, N_moments, max_k), device=device, dtype=cdtype)
    trig_out = torch.zeros((Np, Nu), device=device, dtype=cdtype)
    hk = torch.zeros((N, Nu), device=device, dtype=cdtype) #fourier coefficient_k function (function of u)
    
    X = torch.zeros((N, Np, Nu), device=device, dtype=cdtype)
    
    for k in range(max_k):
        torch.mul(phis2d, k*-1j, out=trig_out)
        torch.exp(trig_out, out=trig_out)
        torch.mul(scaled_sinos, trig_out, out=X)
        torch.sum(X, dim=1, out=hk)
        hk /= 2*torch.pi
        
        res[:, :, k] += torch.einsum("iu,nu->in", hk, normalised_polynomials)

    return res

def _sino_moment_synthesis(coefficients: torch.Tensor, normalised_polynomials: torch.Tensor, W: torch.Tensor, phis2d: torch.Tensor):
    
    device, dtype, cdtype = coefficients.device, normalised_polynomials.dtype, coefficients.dtype
    N, N_moments, max_k = coefficients.shape
    Np, Nu = phis2d.shape

    normalised_polynomials *= W
    normalised_polynomials = normalised_polynomials.to(cdtype) #for einsum :()
    res = torch.zeros((N, Np, Nu), device=device, dtype=dtype)
    basis = torch.zeros((N, Np, Nu), device=device, dtype=cdtype)
    trig_out = torch.zeros((Np, Nu), device=device, dtype=cdtype)

    for k in range(max_k):
        torch.mul(phis2d, k*1j, out=trig_out)
        torch.exp(trig_out, out=trig_out)
        basis[...] = trig_out

        pol_sum = torch.einsum("nu,in->iu", normalised_polynomials, coefficients[:,:,k])[:, None, :]
        if k!= 0:
            pol_sum *= 2 #Acount for conjugate coefficients as well
        basis *= pol_sum
        res += basis.real #this should be real

    return res

def get_moment_mask(coefficients: torch.Tensor):
    "Get the mask corresponding to coefficients that are valid according to the HLCC"
    _, M, K = coefficients.shape
    device = coefficients.device
    ns, ks = torch.arange(0, M, device=device)[:, None].repeat(1, K), torch.arange(0, K, device=device)[None].repeat(M, 1)

    return (ks <= ns) & ((ks + ns) % 2 == 0) #HLCC in basis form

def enforce_moment_constraints(coefficients: torch.Tensor):
    "In place enforcing moment constraints on coefficients. returns reference to input"
    coefficients[:, ~get_moment_mask(coefficients)] *= 0
    return coefficients