import torch


# @torch.jit.script
def _moment_projection(scaled_sinos: torch.Tensor, normalised_polynomials: torch.Tensor, W: torch.Tensor, phis2d: torch.Tensor):
    """
        Faster implementation of the critical step in projection of sinos onto subspace of valid sinograms.
    """
    
    res = scaled_sinos*0

    N_moments, _ = normalised_polynomials.shape
    trig_out = torch.zeros_like(phis2d)
    basis = torch.zeros_like(phis2d)

    X = torch.zeros_like(scaled_sinos)
     
    for n in range(N_moments):    
        pn = normalised_polynomials[n]
        
        for k in range(n%2, n+1, 2):
            if k != 0:
                #Sin basis
                torch.mul(phis2d, k, out=trig_out)
                torch.sin(trig_out, out=trig_out)
                torch.mul(trig_out, pn, out=basis)
                torch.mul(scaled_sinos, basis, out=X)
                projection_coeff = torch.sum(X)
                basis *= projection_coeff / (torch.pi)
                basis *= W
                res += basis
            #Cos basis
            torch.mul(phis2d, k, out=trig_out)
            torch.cos(trig_out, out=trig_out)
            torch.mul(trig_out, pn, out=basis)
            torch.mul(scaled_sinos, basis, out=X)
            projection_coeff = torch.sum(X)
            basis *= projection_coeff / (torch.pi)
            if k == 0: basis /= 2
            basis *= W
            res += basis

    return res


