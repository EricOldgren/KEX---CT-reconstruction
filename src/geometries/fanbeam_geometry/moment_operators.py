import torch
from utils.polynomials import linear_upsample_inside, down_sample, PolynomialBase

"""
will be refactored
"""

def _project_sinos_hlcc(self, sinos: torch.Tensor, PolynomialBasis, N: int, upsample_ratio = 10):
    """
        Project sinos onto subspace of valid sinograms. The infinite basis of this subspace is cutoff for polynomials of degree larger than N.
    """
    us2d = torch.ones_like(self.betas)*linear_upsample_inside(self.us, factor=upsample_ratio) #refine u scale
    Nu_upsampled = us2d.shape[-1]
    betas2d = torch.ones_like(us2d)*self.betas
    scale = self.du*self.db/upsample_ratio

    X = linear_upsample_inside(sinos, factor=upsample_ratio) #lineat interpolation of data
    phis2d = betas2d + torch.arctan(us2d/self.R) - torch.pi/2
    ts2d = us2d*self.R / torch.sqrt(self.R**2 + us2d**2)
    X *= scale
    res = X*0

    polynomials = PolynomialBasis(self.R*self.h/np.sqrt(self.R**2+self.h**2))
    W = polynomials.w(ts2d)

    basis = torch.zeros((N+1, self.Nb, Nu_upsampled), dtype=sinos.dtype, device=sinos.device)
    trig_out = torch.zeros_like(phis2d)
    for n, (pn, l2_norm_n) in enumerate(polynomials.iterate_polynomials(N, ts2d)):
        curr_basis = basis[:n+1]
        pn /= l2_norm_n
        curr_basis[:, :, :] = pn
        
        k, basis_index = n % 2, 0
        while k <= n:
            if k != 0:
                torch.mul(phis2d, k, out=trig_out)
                torch.sin(trig_out, out=trig_out)
                curr_basis[basis_index] *= trig_out
                basis_index += 1
            torch.mul(phis2d, k, out=trig_out)
            torch.cos(trig_out, out=trig_out)
            curr_basis[basis_index] *= trig_out
        


        while False:
            sinb_nk = pn * torch.sin(k*phis2d)
            sinnorm_nk = torch.sum(sinb_nk**2*W*scale)
            cosb_nk = pn * torch.cos(k*phis2d)
            cosnorm_nk = torch.sum(cosb_nk**2*W*scale)

            if k != 0:
                out = torch.einsum("bu,sbu,BU,BU->sBU", sinb_nk, X, W, sinb_nk)
                out /= sinnorm_nk
                res += out
            out = torch.einsum("bu,sbu,BU,BU->sBU", cosb_nk, X, W, cosb_nk)
            out /= cosnorm_nk
            res += out

            k += 2
        
        
        # print("enpoint values", pn[:, 0], pn[:, -1])
    return down_sample(res, factor=upsample_ratio)


