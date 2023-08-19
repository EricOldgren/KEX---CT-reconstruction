import torch
import odl
import numpy as np
from geometries.geometry_base import FBPGeometryBase, DEVICE, DTYPE, CDTYPE, next_power_of_two
from geometries.fanbeam_geometry.moment_operators import MomentProjectionFunction
from utils.polynomials import PolynomialBase, linear_upsample_inside, down_sample_inside, Legendre, Chebyshev, linear_upsample_no_bdry, down_sample_no_bdry

from odl.discr.partition import RectPartition, uniform_partition
from odl.discr.discr_space import DiscretizedSpace, uniform_discr
from odl.tomo.geometry import FanBeamGeometry as odl_fanbeam_geometry
from odl.space.base_tensors import TensorSpace
import odl.contrib.torch as odl_torch


from typing import Type
import matplotlib.pyplot as plt
import time
from statistics import mean




class FlatFanBeamGeometry(FBPGeometryBase):
    """
        Fan Beam Geometry with a flat detector.

        Implementation has functions for forward and backward projections as well as fourier transform along detector axis with appropriate scaling.

        This implementation uses the following notation:
            Beta: rotation angle of the source. This angle is meassured between the central ray and the x-axis and meassured positive counter-clockwise
            u: displacement of a ray along the line orthogonal to the central ray through the origin (center of rotation)
            R: radius of rotation - distance between source and origin
            D: distance between source and detector
            h: max value for u to be on the detector - - i.e h = R / 2D
    """

    def __init__(self, beta_size: int, u_size: int, src_origin: float, src_detector: float, detector_size: float, xy_minmax_bounds: 'tuple[float, float, float, float]', reco_shape: 'tuple[int, int]') -> None:
        """
            Initialize geometry

            Parameters:
                - beta_size (int) : number of angles the source is moved to
                - u_size (int) : number of pixels on the detector
                - src_origin (float) : distance between the source and the center of rotation (origin)
                - src_detector (float) : orthogonal distance between the source and the (flat) detector
                - xy_minmax_bounds (tuple[float]) : (Xmin, Xmax, Ymin, Ymax) - bounding x and y coordinate values for the reco space
                - resco_shape (tuple[int, int]) : (H, W) - shape of reco space images
        """
        super().__init__()


        # Detector parameters
        self.Nb = beta_size
        "number of angles the fan is rotated to"
        self.Nu = u_size
        "resolution of detector"
        self.R = src_origin
        "distance between source and origin, radius of emitter rotation"
        self.D = src_detector
        "distance between source and detector"
        self.h = detector_size * self.R / self.D / 2
        "max u coordinate on a fictive detector through the origin - i.e h = R / 2D"

        self.db = 2*torch.pi / self.Nb
        "distance along beta axis"
        self.betas = torch.linspace(
            0, 2*torch.pi, self.Nb+1, device=DEVICE, dtype=DTYPE)[:-1][:, None]
        "rotations of detector - shape Nb x 1 for convenient broad casting"
        self.du = 2*self.h / self.Nu
        self.us = -self.h+self.du/2 + self.du * \
            torch.arange(0, self.Nu, device=DEVICE, dtype=DTYPE)[None]
        "fictive coordinates of measurements along detector through origin, shape 1 x Nu for convenient broad casting"

        self.jacobian_det = self.R**3 / (self.us**2 + self.R**2)**1.5
        "jacobian determinant for the change of variables from fan coordinates (beta, u) to parallel coordinates (phi, t) - shape  1 x Ny for conveneíent broad casting"

        # Fourier Stuff
        # total size is the nearset power of two two levels up - at most 4 * Ny, at least 2*Ny
        self._fourier_pad_left, self._fourier_pad_right = 0, next_power_of_two(
            self.Nu)*2 - self.Nu
        "number of zeros to pad data with bbefore fourier transform"
        self.ws: torch.Tensor = 2*torch.pi * \
            torch.fft.rfftfreq(self.padded_u_size, d=self.du).to(
                DEVICE, dtype=DTYPE)[None]
        "fourier frequencies the geometry DFT is sampled at. (shape 1 x u_hat_size)"
        self.dw = 2*torch.pi/(self.padded_u_size*self.du)

        # Reconstruction space stuff
        self.NY, self.NX = reco_shape
        xmin, xmax, ymin, ymax = xy_minmax_bounds
        self.dX, self.dY = (xmax - xmin) / self.NX, (ymax - ymin) / self.NY
        "step size in reconstruction space"
        self.Xs = xmin + self.dX / 2 + self.dX * \
            torch.arange(0, self.NX, device=DEVICE, dtype=DTYPE)[None]
        self.Ys = ymin + self.dY/2 + self.dY * \
            torch.arange(0, self.NY, device=DEVICE, dtype=DTYPE)[:, None]
        
        
        vol_space = uniform_discr((xmin, ymin), (xmax, ymax), reco_shape)
        apart = uniform_partition(0, 2*np.pi* (1-1/beta_size), beta_size, nodes_on_bdry=True)
        dpart = uniform_partition(-self.h, self.h, u_size, nodes_on_bdry=False)
        odl_geom = odl_fanbeam_geometry(apart, dpart, src_radius=src_origin, det_radius=0)
        ray_trafo = odl.tomo.RayTransform(vol_space, odl_geom)
        self.Ray = odl_torch.OperatorModule(ray_trafo)
        "Fan Beam Ray transform - Module"
        self.BP = odl_torch.OperatorModule(ray_trafo.adjoint)
        "Fan Beam back projection - Module"

    @property
    def padded_u_size(self):
        return self.Nu + self._fourier_pad_left + self._fourier_pad_right
    @property
    def reco_shape(self):
        "shape in form (Ny, Nx)"
        return (self.NY, self.NX)
    @property
    def n_projections(self):
        "alias of Nb"
        return self.Nb
    @property
    def projection_size(self):
        "alias for Nu"
        return self.Nu

    def fourier_transform(self, sinos: torch.Tensor)->torch.Tensor:
        """
            Returns samples of the fourier transform of a function defined on the detector partition (u-axis).
            Applies torch fft on gpu and scales the result accordingly.
        """
        assert sinos.shape[-1] == self.Nu, "Not an appropriate function"
        ws = self.ws
        sinos = torch.nn.functional.pad(
            sinos, (self._fourier_pad_left, self._fourier_pad_right), "constant", 0)
        # first sampled point in real space
        a = self.us[0, 0] - self.du * self._fourier_pad_left
        return self.du*(torch.cos(a*ws)-1j*torch.sin(a*ws))*torch.fft.rfft(sinos, axis=-1)

    def inverse_fourier_transform(self, sino_hats)->torch.Tensor:
        "Inverse of Geometry.fourier_transform"
        ws = self.ws
        a = self.us[0, 0] - self.du * self._fourier_pad_left
        # Undo padding stuff
        return torch.fft.irfft((torch.cos(a*ws)+1j*torch.sin(a*ws)) / self.du * sino_hats, axis=-1)[:, :, self._fourier_pad_left:-self._fourier_pad_right]

    def project_forward(self, X: torch.Tensor)->torch.Tensor:
        """Radon transform in Fan-Beam coordinates.
            Input X (Tensor) of shape N x NX x NY

            Returns: sinos (Tensor) of shape N x Nb x Nu
        """
        return self.Ray(X).to(DEVICE, dtype=DTYPE)
        # return _project_forward(X, self.Xs, self.Ys, self.betas, self.us, self.R, DEVICE=DEVICE, interpolation_method=0)

    def project_backward(self, X: torch.Tensor)->torch.Tensor:
        "Wegthed BP operator to use for FBP algorithm"
        return self.BP(X)
    
    def ram_lak_filter(self, cutoff_ratio: float = None, full_size = False):
        k = self.ws / (2*torch.pi)
        if cutoff_ratio is not None:
            k[self.ws > self.ws.max()*cutoff_ratio] = 0
        if full_size:
            return k.repeat(self.Nb, 1)
        return k

    def fbp_reconstruct(self, sinos: torch.Tensor):
        "reconstruct sinos using FBP"
        return self.project_backward(self.inverse_fourier_transform(self.fourier_transform(sinos*self.jacobian_det)*self.ram_lak_filter()/2))

    def reflect_fill_sinos(self, sinos: torch.Tensor, known_beta_bools: torch.Tensor, linear_interpolation = False):
        """
            in place flling of sinogram
            applied on full 360deg sinograms, fills unknown region of sinogram by finding equivalent lines on opposite side
        """
        assert known_beta_bools.shape == (self.Nb,)
        Nunknown = int((~known_beta_bools).sum())
        unknown_betas = self.betas[~known_beta_bools].repeat(1, self.Nu) #shape Nunknonw x Nu
        unknown_alphas = torch.arctan(self.us / self.R).repeat(Nunknown, 1) # shape Nunknown x Nu

        reflected_betas = unknown_betas + torch.pi - 2*unknown_alphas
        u_inds = torch.arange(self.Nu-1, -1, -1, device=DEVICE)[None, :].repeat(Nunknown, 1) #flipped order as angle is opposite sign

        if linear_interpolation:
            beta_inds_lower = (reflected_betas / self.db).to(dtype=torch.int64)
            beta_inds_upper = beta_inds_lower + 1
            beta_inds_lower[beta_inds_lower >= self.Nb] -= self.Nb
            beta_inds_upper[beta_inds_upper >= self.Nb] -= self.Nb
            beta_weights_upper = reflected_betas.frac()
            beta_weights_lower = 1 - beta_weights_upper
            
            sinos[:, ~known_beta_bools] = sinos[:, beta_inds_lower, u_inds]*beta_weights_lower + sinos[:, beta_inds_upper, u_inds]*beta_weights_upper #Linear interpolation
        else:
            beta_inds = (reflected_betas / self.db + 0.5).to(dtype=torch.int64)
            beta_inds[beta_inds>=self.Nb] -= self.Nb
            sinos[:, ~known_beta_bools] = sinos[:, beta_inds, u_inds] ##NN interpolation

        return sinos

    def zero_cropp_sinos(self, sinos: torch.Tensor, ar: float, start_ind: int):
        """
            Cropp sinograms to limited angle data. Sinos are set to zero outside cropped region

            return cropped_sinos, known_beta_bool
        """
        n_projs = int(self.n_projections * ar)
        end_ind = (start_ind + n_projs) % self.n_projections
        known = torch.zeros(self.Nb, dtype=bool, device=DEVICE)
        if start_ind < end_ind:
            known[start_ind:end_ind] = True
        else:
            known[start_ind:] = True
            known[:end_ind] = True
        res = sinos*0
        res[:, known, :] = sinos[:, known, :]

        return res, known

    def rotate_sinos(self, sinos: torch.Tensor, shift: int):
        """
            shift sinos in cycle by shift steps
        """

        return torch.concat([
            sinos[:, -shift:, :], sinos[:, :-shift, :] #works for shift positive and negative
        ], dim=1) 

    
    def project_sinos(self, sinos: torch.Tensor, PolynomialBasis: Type[PolynomialBase], N: int, upsample_ratio = 1):
        """
            Project sinos onto subspace of valid sinograms. The infinite basis of this subspace is cutoff for polynomials of degree larger than N.
        """
        us_upsampled = linear_upsample_no_bdry(self.us, factor=upsample_ratio) #refine u scale
        X = linear_upsample_no_bdry(sinos, factor=upsample_ratio) #lineat interpolation of data
        betas2d = torch.ones_like(X[0])*self.betas
        scale = self.du*self.db/upsample_ratio * self.R**3 / (us_upsampled**2 + self.R**2)**1.5 #volume element per sinogram cell
        X *= scale

        phis2d = betas2d + torch.arctan(us_upsampled/self.R) - torch.pi/2
        ts1d = (us_upsampled*self.R / torch.sqrt(self.R**2 + us_upsampled**2))[0]

        polynomials = PolynomialBasis(self.R*self.h/np.sqrt(self.R**2+self.h**2))
        W = polynomials.w(ts1d)
        normalised_polynomials = torch.stack([pn / np.sqrt(l2_norsm_sq) for pn, l2_norsm_sq in polynomials.iterate_polynomials(N, ts1d)])

        return down_sample_no_bdry(
            MomentProjectionFunction.apply(X, normalised_polynomials, W, phis2d),
            # _moment_projection(X, normalised_polynomials, W, phis2d),
            factor=upsample_ratio
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    phantoms = torch.stack(torch.load("data/HTC2022/HTCTestPhantomsFull.pt", map_location=DEVICE))[:2]
    # phantoms = torch.load("data/kits_phantoms_256.pt", map_location=DEVICE)[:1, 1]
    print(phantoms.shape)

    geometry = FlatFanBeamGeometry(720, 560, 410.66, 543.74, 112, [-40,40, -40, 40], [512, 512])
    # geometry = FlatFanBeamGeometry(700, 560, 10.0, 14.0, 3.0, [-1.0,1.0, -1.0, 1.0], [256, 256])
    start = time.time()
    sinos = geometry.project_forward(phantoms)
    print("fprward projection took", time.time()-start, "s")
    la_sinos, known_beta_bools = geometry.zero_cropp_sinos(sinos, 0.6, 0)

    start = time.time()
    print("beginning orthogonal projection")

    start = time.time()
    projected_sinos = geometry.project_sinos(sinos, Legendre, 200, 1)
    print("projection took", time.time()-start, "s")
    print("sino mse", torch.mean((projected_sinos-sinos)**2))

    recons = geometry.fbp_reconstruct(sinos)
    recons_projected = geometry.fbp_reconstruct(projected_sinos)
    print("recon mse", torch.mean((recons_projected-recons)**2))

    inspect_ind = 0
    fig, _ = plt.subplots(1,3)
    plt.subplot(131)
    plt.imshow(sinos[inspect_ind].cpu())
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(projected_sinos[inspect_ind].cpu())
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(torch.abs(projected_sinos[inspect_ind]-sinos[inspect_ind]))
    plt.colorbar()
    fig.show()
    
    fig, _ = plt.subplots(1,2)
    plt.subplot(121)
    plt.imshow(recons[inspect_ind].cpu())
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(recons_projected[inspect_ind].cpu())
    plt.colorbar()
    fig.show()

    plt.show()


#Deprecated
# def _slow_sino_projection(self, sinos: torch.Tensor, PolynomialBasis: Type[PolynomialBase], N: int, upsample_ratio = 11):
#         """
#             Project sinos onto subspace of valid sinograms. The infinite basis of this subspace is cutoff for polynomials of degree larger than N. This is slower than project sinos
#         """
#         us_upsampled = linear_upsample_no_bdry(self.us, factor=upsample_ratio) #refine u scale
#         Nu_upsampled = us_upsampled.shape[-1]
#         us2d = torch.ones_like(self.betas)*us_upsampled
#         betas2d = torch.ones_like(us2d)*self.betas
#         scale = self.du*self.db/upsample_ratio * self.R**3 / (us_upsampled**2 + self.R**2)**1.5 #volume element per sinogram cell

#         X = linear_upsample_no_bdry(sinos, factor=upsample_ratio) #lineat interpolation of data
#         phis2d = betas2d + torch.arctan(us2d/self.R) - torch.pi/2
#         ts2d = us2d*self.R / torch.sqrt(self.R**2 + us2d**2)
#         X *= scale
#         res = X*0

#         polynomials = PolynomialBasis(self.R*self.h/np.sqrt(self.R**2+self.h**2))
#         W = polynomials.w(ts2d)

#         trig_out = torch.zeros_like(phis2d)
#         for n, (pn, l2_normsq_n) in enumerate(polynomials.iterate_polynomials(N, ts2d)):
#             k, basis_index = n % 2, 0
            
#             # curr_basis = pn.repeat(n+1, 1, 1)
#             # while k <= n:
#             #     if k != 0:
#             #         torch.mul(phis2d, k, out=trig_out)
#             #         torch.sin(trig_out, out=trig_out)
#             #         curr_basis[basis_index] *= trig_out
#             #         basis_index += 1
#             #     torch.mul(phis2d, k, out=trig_out)
#             #     torch.cos(trig_out, out=trig_out)
#             #     curr_basis[basis_index] *= trig_out
#             #     basis_index += 1
            
#             #     res += torch.einsum("nub, sub, UB, nUB->sUB", curr_basis, X, W, curr_basis)
#             print("projecting onto polynomials of degree", n)
#             print("Norms of basis functions:")
#             while k <= n:
#                 sinb_nk = pn * torch.sin(k*phis2d)
#                 sinnorm_nk = torch.sum(sinb_nk**2*W*scale)
#                 cosb_nk = pn * torch.cos(k*phis2d)
#                 cosnorm_nk = torch.sum(cosb_nk**2*W*scale)
#                 print("\t", n, k, "sin num", sinnorm_nk, "num / analytic:", sinnorm_nk/(l2_normsq_n*torch.pi))
#                 print("\t", n, k, "cos num", cosnorm_nk, "num / analytic:", cosnorm_nk/(l2_normsq_n*(2*torch.pi if k == 0 else torch.pi)))

#                 if k != 0:
#                     out = torch.einsum("bu,sbu,BU,BU->sBU", sinb_nk, X, W, sinb_nk)
#                     out /= sinnorm_nk
#                     res += out
#                 out = torch.einsum("bu,sbu,BU,BU->sBU", cosb_nk, X, W, cosb_nk)
#                 out /= cosnorm_nk
#                 res += out

#                 k += 2
        
#         return down_sample_no_bdry(res, factor=upsample_ratio)
