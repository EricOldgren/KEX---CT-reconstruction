import torch
import odl
import numpy as np
from utils.geometries.geometry_base import FBPGeometryBase, DEVICE, DTYPE, nearest_power_of_two

from odl.discr.partition import RectPartition, uniform_partition
from odl.discr.discr_space import DiscretizedSpace, uniform_discr
from odl.tomo.geometry import FanBeamGeometry as odl_fanbeam_geometry
from odl.space.base_tensors import TensorSpace
import odl.contrib.torch as odl_torch


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
        self.betas = torch.linspace(
            0, 2*torch.pi, self.Nb+1, device=DEVICE, dtype=DTYPE)[:-1][:, None]
        "rotations of detector - shape Nb x 1 for convenient broad casting"
        self.du = 2*self.h / self.Nu
        self.us = -self.h+self.du/2 + self.du * \
            torch.arange(0, self.Nu, device=DEVICE, dtype=DTYPE)[None]
        "fictive coordinates of measurements along detector through origin, shape 1 x Ny for convenient broad casting"

        self.jacobian_det = self.R**3 / (self.us**2 + self.R**2)**1.5
        "jacobian determinant for the change of variables from fan coordinates (beta, u) to parallel coordinates (phi, t) - shape  1 x Ny for conveneíent broad casting"

        # Fourier Stuff
        # total size is the nearset power of two two levels up - at most 4 * Ny, at least 2*Ny
        self._fourier_pad_left, self._fourier_pad_right = 0, nearest_power_of_two(
            self.Nu)*2 - self.Nu
        "number of zeros to pad data with bbefore fourier transform"
        self.ws: torch.Tensor = 2*torch.pi * \
            torch.fft.rfftfreq(self.padded_u_size, d=self.du).to(
                DEVICE, dtype=DTYPE)[None]
        "fourier frequencies the geometry DFT is sampled at."
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
        apart = uniform_partition(0, 2*np.pi, beta_size, nodes_on_bdry=True)
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
    def u_size(self):
        return self.Nu
    @property
    def beta_size(self):
        return self.Nb

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
        return self.Ray(X)
        # return _project_forward(X, self.Xs, self.Ys, self.betas, self.us, self.R, DEVICE=DEVICE, interpolation_method=0)

    def project_backward(self, X: torch.Tensor)->torch.Tensor:
        "Wegthed BP operator to use for FBP algorithm"
        return self.BP(X)
    
    def ram_lak_filter(self, cutoff_ratio: float = None):
        "Ram-Lak filter in frequency domain"
        return self.ws / (2*torch.pi)


    def fbp_reconstruct(self, sinos: torch.Tensor):
        "reconstruct sinos using FBP"
        return self.BP(self.inverse_fourier_transform(self.fourier_transform(sinos)*self.ram_lak_filter()/2))

def plot_hepler(data: torch.Tensor, points: torch.Tensor, dX: torch.Tensor, dY: torch.Tensor, ping_val = 100):
    import matplotlib.pyplot as plt
    plot_data = torch.tensor(data[0])
    n, _ = points.shape
    inds_X, inds_Y = (points[:, 0] / dX).to(int), (points[:, 1] / dY).to(int)
    for i in range(n):
        px = inds_X[i]
        py = inds_Y[i]
        plot_data[py-10:py+10, px-10:px+10] = ping_val


    plt.subplot(121)
    plt.imshow(plot_data)
    plt.subplot(122)
    plt.scatter(points[:, 0], points[:, 1])
    plt.show()

def draw_points_and_grid(Xmin, Xmax, Ymin, Ymax, points, *plot_pointss):
    import matplotlib.pyplot as plt
    plt.plot([Xmin]*40, torch.linspace(Ymin, Ymax, 40), c="r")
    plt.plot([Xmax]*40, torch.linspace(Ymin, Ymax, 40), c="r")
    plt.plot(torch.linspace(Xmin, Xmax, 40), [Ymin]*40, c="r")
    plt.plot(torch.linspace(Xmin, Xmax, 40), [Ymax]*40, c="r")

    for plot_points in plot_pointss:
        plt.plot(plot_points[:, 0], plot_points[:, 1], c="g")


    plt.scatter(points[:, 0], points[:, 1], c="b")

    plt.show()
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    phantoms = torch.stack(torch.load("data/HTC2022/HTCTestPhantomsFull.pt", map_location=DEVICE))
    # phantoms = torch.load("data/kits_phantoms_256.pt", map_location=DEVICE)[:500, 0]
    print(phantoms.shape)

    geometry = FlatFanBeamGeometry(720, 560, 410.66, 543.74, 112, [-40,40, -40, 40], [512, 512])
    # geometry = FlatFanBeamGeometry(700, 560, 6.0, 10.0, 2.0, [-1.0,1.0, -1.0, 1.0], [256, 256])
    # plt.imshow(phantoms[0].cpu())
    # plt.show()
    times = []
    for _ in range(20):
        start = time.time()
        sinos = geometry.project_forward(phantoms)
        times.append(time.time() - start)
        print("projection took", times[-1], "s")
    print("="*40)
    print("Average projection time", mean(times))
    plt.imshow(sinos.cpu().numpy()[2])
    plt.colorbar()
    plt.show()

    print("hello")

