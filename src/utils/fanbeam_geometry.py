import torch
import numba

#DEBUG
import matplotlib.pyplot as plt
import time
#DEBUG


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float
CDTYPE = torch.cfloat
eps = torch.finfo(DTYPE).eps

def nearest_power_of_two(n: int):
    P = 1
    while P < n:
        P *= 2
    return P


class FlatFanBeamGeometry:
    """
        Fan Beam Geometry with a flat detector.

        Implementation has functions for forward and backward projections as well as fourier transform along detector axis with appropriate scaling.
        
        Everything is based solely on pytorch.

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
            torch.arange(1, self.Nu, device=DEVICE, dtype=DTYPE)[None]
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
                DEVICE, dtype=DTYPE)
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
        a = self.us[0] - self.du * self._fourier_pad_left
        return self.du*(torch.cos(a*ws)-1j*torch.sin(a*ws))*torch.fft.rfft(sinos, axis=-1)

    def inverse_fourier_transform(self, sino_hats)->torch.Tensor:
        "Inverse of Geometry.fourier_transform"
        ws = self.ws
        a = self.us[0] - self.du * self._fourier_pad_left
        # Undo padding stuff
        return torch.fft.irfft((torch.cos(a*ws)+1j*torch.sin(a*ws)) / self.du * sino_hats, axis=-1)[:, :, self._fourier_pad_left:-self._fourier_pad_right]

    def project_forward(self, X: torch.Tensor):
        """Radon transform in Fan-Beam coordinates.
            Input X (Tensor) of shape N x NX x NY

            Returns: sinos (Tensor) of shape N x Nb x Nu
        """
        return _project_forward(X, self.Xs, self.Ys, self.betas, self.us, self.R, DEVICE=DEVICE)

    def project_backward(self, X: torch.Tensor):
        "Wegthed BP operator to use for FBP algorithm"
        raise NotImplementedError()

    def fbp_reconstruct(self, sinos: torch.Tensor):
        "reconstruct sinos using FBP"
        raise NotImplementedError()

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
    

torch.jit.enable_onednn_fusion(True)
@torch.jit.script
def _project_forward(data: torch.Tensor, Xs: torch.Tensor, Ys: torch.Tensor, betas: torch.Tensor, us: torch.Tensor, R: float, DEVICE: torch.device):
    
    data = torch.flip(data, dims=(1,)) #flip y upwards
    
    N_samples, _, _ = data.shape
    Xs, Ys = Xs.reshape(-1), Ys.reshape(-1)
    betas, us = betas.reshape(-1, 1), us.reshape(1, -1)
    Nb, _ = betas.shape
    _, Nu = us.shape
    Xmin, Xmax, Ymin, Ymax = Xs[0], Xs[-1], Ys[0], Ys[-1]
    dX, dY = torch.mean(Xs[1:]-Xs[:-1]), torch.mean(Ys[1:] - Ys[:-1])
    min_d = torch.minimum(dX, dY)

    N_line_points = (((Xmax - Xmin)**2 + (Ymax-Ymin)**2)**0.5 / min_d).to(dtype=torch.int32)
    ratios = (1/(2*N_line_points) + torch.arange(0,N_line_points, device=DEVICE, dtype=torch.float32)/N_line_points)

    # Define lines of Rays
    Sources = torch.stack([torch.cos(betas), torch.sin(betas)],dim=-1).reshape(-1,1,2)*R
    ProjectorPoints = torch.stack([torch.sin(betas), -torch.cos(betas)], dim=-1).reshape(-1,1,2) * us.reshape(1,-1,1)
    line_directions = ProjectorPoints - Sources
    line_directions /= torch.linalg.norm(line_directions)
    line_normals = torch.stack([line_directions[:, :, 1], -line_directions[:, :, 0]], dim=-1)
    line_translations = torch.sum(ProjectorPoints*line_normals, dim=-1, keepdim=True)
    
    # Find endpoints of lines in region
    bounding_normals = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], device=DEVICE, dtype=torch.float32) #edges
    bounding_vals = torch.stack([Xmin, -Xmax, Ymin, -Ymax])
    A = torch.stack([
        bounding_normals[None, None].repeat(Nb, Nu, 1, 1), line_normals[:,:,None].repeat(1,1,4,1)
    ], dim=3)
    ts = torch.stack([
        bounding_vals[None, None, :, None].repeat(Nb, Nu, 1, 1), line_translations[:,:,None].repeat(1,1,4,1)
    ], dim=3)
    intersections: torch.Tensor = torch.linalg.solve(A, ts).reshape(Nb, Nu, 4, 2) #Solve for intersection of rays and edges
    on_inside = ((torch.einsum("nk,buik->buin", bounding_normals, intersections) - bounding_vals) > -1e-5).sum(dim=-1) == 4
    invalids = (on_inside.sum(dim=-1) != 2)
    on_inside[invalids, :] = torch.tensor([1,1,0,0], device=DEVICE, dtype=torch.bool)
    intersections = intersections[on_inside].view(Nb, Nu, 2, 2)
    #Define line points to integrate over
    start_points, stop_points = intersections[:, :, 0]*0, intersections[:,:,0]*0
    zero2one = torch.sum((intersections[:,:, 1]-intersections[:,:,0])*line_directions, dim=-1) > 0.0 #orientations where line_dir goes from int[0] - int[1]
    start_points[zero2one, :] += intersections[zero2one][:, 0]; stop_points[zero2one, :] += intersections[zero2one][:, 1]
    start_points[~zero2one, :] += intersections[~zero2one][:, 1]; stop_points[~zero2one, :] += intersections[~zero2one][:, 0]
    line_segments = stop_points - start_points
    dls = torch.linalg.norm(line_segments, dim=-1, keepdim=True)
    dls /= N_line_points
    dls[invalids] *= 0

    line_points: torch.Tensor = ratios.reshape(1,1,-1,1)*line_segments[...,None, :]
    line_points += start_points[...,None, :]
    line_points[invalids] *= 0
    #Find inds of line points
    line_points -= torch.stack([Xmin, Ymin])
    line_points /= torch.stack([dX, dY])
    line_points += 0.5
    inds = line_points.to(torch.int64) #integer inds

    res = torch.zeros((N_samples, Nb, Nu), device=DEVICE, dtype=betas.dtype)
    for ind in range(N_samples):
        res[ind] += torch.einsum("ubl,ubl->ub", data[ind, inds[...,1], inds[...,0]], dls) # this uses nearest neighbour interpolation - faster

    return res


    #Code for linear interpolation
    # wxs, wys = (line_points[...,0] - Xs[inds_X]) / dX, (line_points[..., 1] - Ys[inds_Y]) / dY
    # return  torch.sum(
    #         data[:,H-1-inds_Y, inds_X]*(1-wxs)*(1-wys)*dls +
    #         data[:,H-1-(inds_Y+1), inds_X] * wxs * (1-wys)*dls +
    #         data[:,H-1-inds_Y, torch.minimum(inds_X+1, torch.tensor(Xs.shape[0]-1))]*(1-wxs)*wys*dls +
    #         data[:,H-1-(inds_Y+1), torch.minimum(inds_X+1, torch.tensor(Xs.shape[0]-1))] * wxs*wys*dls, dim=-1
    #     )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # phantoms = torch.stack(torch.load("data/HTC2022/HTCTestPhantomsFull.pt"))
    phantoms = torch.load("data/kits_phantoms_256.pt")[:500, 0]
    print(phantoms.shape)

    # geometry = FlatFanBeamGeometry(720, 560, 410.66, 543.74, 112, [-40,40, -40, 40], [512, 512])
    geometry = FlatFanBeamGeometry(700, 560, 6.0, 10.0, 2.0, [-1.0,1.0, -1.0, 1.0], [256, 256])
    # plt.imshow(phantoms[0].cpu())
    # plt.show()
    start = time.time()
    sinos = geometry.project_forward(phantoms)
    print("projection took", time.time()-start, "s")
    plt.imshow(sinos.cpu().numpy()[2])
    plt.colorbar()
    plt.show()

    print("hello")

