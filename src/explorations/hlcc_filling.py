import torch

from utils.tools import DEVICE, DTYPE, CDTYPE
from utils.polynomials import Legendre
from geometries import FBPGeometryBase, get_moment_mask

def conjugate_gradient_solver(la_sinos: torch.Tensor, ar: float, geometry: FBPGeometryBase, M: int, K: int, PolynomialFamily = Legendre, tol = 0.1):
    batch_size = la_sinos.shape[0]
    n_known = geometry.n_known_projections(ar)
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
        exp[:, n_known:] = 0
        w = geometry.series_expand(exp, PolynomialFamily, M, K)[:, mask] #Apk -  shape batch_size x n

        ak = (rk*rk.conj()).sum(dim=-1, keepdim=True) / (pk.conj()*w).sum(dim=-1, keepdim=True) # shape batch_size x 1
        cnext = ck + ak*pk
        #Calculate r
        embedding[:, mask] = cnext
        exp = geometry.synthesise_series(embedding, PolynomialFamily)
        exp[:, n_known:] = 0
        rnext = geometry.series_expand(la_sinos-exp, PolynomialFamily, M, K)[:, mask]
        # rnext = rk - ak*w

        bk = (rnext.conj()*rnext).sum(dim=-1, keepdim=True) / (rk*rk.conj()).sum(dim=-1, keepdim=True) #shape: bacth_size x 1
        pnext = rnext + bk*pk

        ck = cnext
        rk = rnext
        pk = pnext
        k += 1

        ##Debug INFO
        
        print("k:", k, "res mse:", torch.linalg.norm(rk, dim=-1))#, "absolute mse:", torch.linalg.norm(err, dim=-1))
        ###
        

    embedding[:, mask] = ck
    return geometry.synthesise_series(embedding, PolynomialFamily)


if __name__ == "__main__":
    from geometries import HTC2022_GEOMETRY
    from utils.tools import MSE
    from utils.data import get_htc2022_train_phantoms
    import matplotlib.pyplot as plt


    geometry = HTC2022_GEOMETRY
    phantoms = get_htc2022_train_phantoms()
    print("phantoms loaded")
    ar = 0.25
    M, K = 64, 64
    print("calculcating sinos")
    sinos = geometry.project_forward(phantoms)
    print("sinos calculated")
    la_sinos, known_angles = geometry.zero_cropp_sinos(sinos, ar, 0)

    exp = conjugate_gradient_solver(la_sinos, ar, geometry, M, K, Legendre)

    print("Error:", MSE(exp, sinos))

    disp_ind = 2
    plt.subplot(121)
    plt.imshow(exp[disp_ind].cpu())
    plt.title("exp")
    plt.subplot(122)
    plt.imshow(sinos[disp_ind].cpu())
    plt.title("gt")

    plt.show()

