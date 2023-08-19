from typing import Any
import torch

"""¨
TODO:
integrate _moment_projectio with pytorch autograd. Orthogonal projection is hermitian so back propagation uses same operator as forward propagator
"""


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

