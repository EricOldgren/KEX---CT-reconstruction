from typing import Tuple
import torch

from geometries.geometry_base import FBPGeometryBase, next_power_of_two, DEVICE, DTYPE, CDTYPE


from odl import Operator
from odl.tomo import RayTransform as odl_RayTransform, Parallel2dGeometry as odl_parallel_geometry
from odl.discr.discr_space import uniform_discr
from odl.discr.partition import uniform_partition
import odl.contrib.torch as odl_torch

import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt


class ParallelGeometry(FBPGeometryBase):
    """
        Parallel 2D Geometry.
        Implementation has functions for forward and backward projections as well as fourier transform along detector axis with appropriate scaling and projection of sinograms onto subspace of valid sinograms..

        This implementation uses the following notation:
            phi: angle bewteen normal of line and the x-axis, meassured positive counter-clockwise (thus phi = 0 is lines pointing straight up)
            t: displacement of line
            rho: radius of reco space and max value along detector
    """
    
    def __init__(self, phi_size: int, t_size: int, xy_minmax_bounds: 'tuple[float, float, float, float]', reco_shape: 'tuple[int, int]'):
        
        super().__init__()
        self._init_args = (phi_size, t_size, xy_minmax_bounds, reco_shape)
        self.Np, self.Nt = phi_size, t_size

        # Reconstruction space stuff
        self.NY, self.NX = reco_shape
        xmin, xmax, ymin, ymax = xy_minmax_bounds
        self.dX, self.dY = (xmax - xmin) / self.NX, (ymax - ymin) / self.NY
        "step size in reconstruction space"
        self.Xs = xmin + self.dX / 2 + self.dX * \
            torch.arange(0, self.NX, device=DEVICE, dtype=DTYPE)[None]
        self.Ys = ymin + self.dY/2 + self.dY * \
            torch.arange(0, self.NY, device=DEVICE, dtype=DTYPE)[:, None]
        
        #Sino space stuff
        self.rho = torch.linalg.norm(torch.tensor([xmax-xmin, ymax-ymin], dtype=DTYPE)).item()
        self.dphi, self.dt = 2*torch.pi / self.Np, 2*self.rho/ self.Nt
        self.phis = torch.linspace(
            0, 2*torch.pi, self.Np+1, device=DEVICE, dtype=DTYPE)[:-1][:, None]
        self.ts = (-self.rho + self.dt*torch.arange(0, self.Nt, device=DEVICE, dtype=DTYPE))[None]
        self.jacobian_det = torch.ones_like(self.ts)
        "jacobian det - trivial in parallel geometry i.e tensor of ones"        

        #Fourier domain stuff
        self._fourier_pad_left, self._fourier_pad_right = 0, next_power_of_two(t_size)*2 - t_size #total size is the nearset power of two two levels up - at most 4 * t_size
        self.ws: torch.Tensor = 2*torch.pi * \
            torch.fft.rfftfreq(self.padded_t_size, d=self.dt).to(
                DEVICE, dtype=DTYPE)[None]
        "fourier frequencies the geometry DFT is sampled at. (shape 1 ws_size)"      
        self.omega: float = torch.pi * min(1.0 / (self.dphi*self.rho), 1 / self.dt)
        "maximum bandwidth for functions to be exactly restored from this sinogram sampling scheme."
    
        # Make a parallel beam geometry with odl
        apart = uniform_partition(0, 2*torch.pi* (1-1/phi_size), phi_size, nodes_on_bdry=True)
        dpart = uniform_partition(-self.rho, self.rho, t_size, nodes_on_bdry=False)
        odl_geom = odl_parallel_geometry(apart, dpart)
        vol_space = uniform_discr((xmin, ymin), (xmax, ymax), reco_shape)
        ray_trafo = odl_RayTransform(vol_space, odl_geom)

        self.Ray = odl_torch.OperatorModule(ray_trafo)
        "Parallel Beam Ray transform - Module"
        self.BP = odl_torch.OperatorModule(ray_trafo.adjoint)
        "Parallel Beam backprojection - Module"
    def get_init_args(self):
        return self._init_args

    @property
    def n_projections(self):
        return self.Np
    @property
    def projection_size(self):
        return self.Nt
    @property
    def padded_t_size(self):
        return self._fourier_pad_left + self.Nt + self._fourier_pad_right
   
    
    def fourier_transform(self, sinos: torch.Tensor):
        """
            Returns samples of the fourier transform of a function defined on the detector partition.
            Applies the torch fft on gpu and scales the result accordingly.
        """
        assert sinos.shape[-1] == self.Nt, "Not an appropriate function"
        a = -self.rho - self.dt * self._fourier_pad_left   #first sampled point in real space
        
        sinos = torch.nn.functional.pad(sinos, (self._fourier_pad_left, self._fourier_pad_right), "constant", 0)
        return self.dt*(torch.cos(a*self.ws)-1j*torch.sin(a*self.ws))*torch.fft.rfft(sinos, axis=-1) #self.dt*torch.exp(-1j*a*self.fourier_domain)*torch.fft.rfft(sino, axis=-1)
    
    def inverse_fourier_transform(self, sino_hats, padding = False):
        "Inverse of Geometry.fourier_transform"
        a = -self.rho - self.dt * self._fourier_pad_left
        back_scaled = (torch.cos(a*self.ws)+1j*torch.sin(a*self.ws)) / self.dt * sino_hats # torch.exp(1j*a*self.fourier_domain) / self.dt * sino_hat
        sinos = torch.fft.irfft(back_scaled, axis=-1)
        sinos = sinos[:, :, self._fourier_pad_left:-self._fourier_pad_right] #undo padding
        return sinos
    
    def project_forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.Ray(X).to(DEVICE, dtype=DTYPE)
    def project_backward(self, X: torch.Tensor) -> torch.Tensor:
        return self.BP(X)
    
    def ram_lak_filter(self, cutoff_ratio: float = None, full_size=False) -> torch.Tensor:
        k = self.ws.to(CDTYPE) / (2*torch.pi)
        if cutoff_ratio is not None:
            k[self.ws > self.ws.max()*cutoff_ratio] = 0
        if full_size:
            return k.repeat(self.Np, 1)
        return k

    def fbp_reconstruct(self, sinos: torch.Tensor) -> torch.Tensor:
        return self.project_backward(self.inverse_fourier_transform(self.fourier_transform(sinos)*self.ram_lak_filter()/2))
    
    def zero_cropp_sinos(self, sinos: torch.Tensor, ar: float, start_ind: int) -> Tuple[torch.Tensor, torch.Tensor]:
        n_projs = int(self.n_projections * ar)
        end_ind = (start_ind + n_projs) % self.n_projections
        known = torch.zeros(self.n_projections, dtype=bool, device=DEVICE)
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
    
    def reflect_fill_sinos(self, sinos: torch.Tensor, known_beta_bools: torch.Tensor, linear_interpolation=False):
        """In place flling of sinogram applied on full 360deg sinograms. In parallel geometry this is exact as long as phi_size is even, thus no interpolation is done.
        """
        inds = torch.arange(0, self.Np, device=DEVICE)
        unknown_inds = inds[~known_beta_bools]
        reflected_inds = (unknown_inds + self.n_projections//2)
        reflected_inds %= self.n_projections
        sinos[..., unknown_inds, :] = torch.flip(sinos[..., reflected_inds, :], dims=(-1,))
        return sinos
    
    def __repr__(self) -> str:
        return f"""Geometry(
            angle ratio: {self.ar} phi_size: {self.Np} t_size: {self.Nt}
            reco_space: {self.reco_space}
        )"""


if __name__ == "__main__":
    PHANTOMS = torch.stack(torch.load("data/HTC2022/HTCTestPhantomsFull.pt")).to(DEVICE, dtype=DTYPE)
    
    geometry = ParallelGeometry(900, 300, [-1,1,-1,1], [512, 512])
    ar = 0.5
    SINOS = geometry.project_forward(PHANTOMS)
    sinos_la, known_beta_bools = geometry.zero_cropp_sinos(SINOS, ar, 0)
    geometry.reflect_fill_sinos(sinos_la, known_beta_bools)

    recons = geometry.fbp_reconstruct(sinos_la)
    recons_orig = geometry.fbp_reconstruct(SINOS)

    print("="*40)
    print("Sino max absolute error:", torch.max(torch.abs(sinos_la-SINOS)))
    print("recon mse:", torch.mean((recons-PHANTOMS)**2))
    print("recon mse orig:", torch.mean((recons_orig-PHANTOMS)**2))
    print("="*40)

    inspect_ind = 3

    fig, _ = plt.subplots(1,2)
    plt.subplot(121)
    plt.imshow(SINOS[inspect_ind].cpu())
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(sinos_la[inspect_ind].cpu())
    plt.colorbar()
    fig.show()

    fig, _ = plt.subplots(1,3)
    plt.subplot(131)
    plt.imshow(PHANTOMS[inspect_ind].cpu())
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(recons[inspect_ind].cpu())
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(recons_orig[inspect_ind].cpu())
    plt.colorbar()
    fig.show()

    plt.show()    






