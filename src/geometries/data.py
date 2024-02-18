import torch
import torchvision
import numpy as np
import scipy
from sklearn.model_selection import train_test_split
from typing import Tuple, Union
import logging

from torch.autograd.functional import hessian
from utils.tools import DEVICE, DTYPE, no_bdry_linspace, GIT_ROOT
from geometries.fanbeam_geometry import FlatFanBeamGeometry

##Geometry Config
scale_factor = 1.0
htc_th = 0.02 / scale_factor
htc_sino_var = 2e-2 / scale_factor
htc_mean_attenuation = 0.032 / scale_factor
htc_nprojections_by_level = [181, 161, 141, 121, 101, 81, 61]
HTC2022_GEOMETRY = FlatFanBeamGeometry(720, 560, 410.66*scale_factor, 543.74*scale_factor, 112.0*scale_factor, [-38*scale_factor,38*scale_factor, -38*scale_factor,38*scale_factor], [512, 512])


#Data loading
def get_htc2022_train_phantoms():
    return torch.stack(torch.load( GIT_ROOT / "data/HTC2022/HTCTrainingPhantoms.pt", map_location=DEVICE)).to(DTYPE)
def get_synthetic_htc_phantoms(use_kits=False):
    "retrieve generated phantoms phantoms concatenated with the kits data set"
    generated = torch.load(GIT_ROOT / "data/synthetic_htc_bigbatch.pt", map_location=DEVICE).to(DTYPE)
    if not use_kits:
        return generated
    kits = get_kits_train_phantoms(resize=True)
    kits *= htc_mean_attenuation / 2 #mean value of phantoms is more than one
    return torch.concat([generated, kits])


def get_kits_train_phantoms(resize = True)->torch.Tensor:
    if resize:
        return torchvision.transforms.functional.resize(torch.load(GIT_ROOT / "data/kits_phantoms_256.pt", map_location=DEVICE)[:, 1], (512, 512))
    return torch.load(GIT_ROOT / "data/kits_phantoms_256.pt", map_location=DEVICE)[:, 1]

def get_htc_traindata():
    "return sinos, phantoms"
    sinos = torch.stack(torch.load(GIT_ROOT / "data/HTC2022/HTCTrainingData.pt", map_location=DEVICE))[:, :720].to(DTYPE)
    phantoms = HTC2022_GEOMETRY.fbp_reconstruct(sinos)

    return sinos, phantoms
def get_htc_testdata(level: int):
    "return sinos, known_angles, to_rotate, phantoms"
    assert level >= 1 and level <= 7, "invalid level {level}"
    n_known_angles = [181, 161, 141, 121, 101, 81, 61]

    sinos = []
    to_rotate = []
    phantoms = []

    for c in ['a','b','c']:
        phantom = torch.tensor(scipy.io.loadmat(GIT_ROOT/('data/HTC2022/TestData/htc2022_0' + str(level) + c + '_recon_fbp_seg.mat'))['reconFullFbpSeg']).to(DEVICE, dtype=bool)
        phantoms.append(phantom)
        
        dataLimited = scipy.io.loadmat(GIT_ROOT/('data/HTC2022/TestData/htc2022_0' + str(level) + c + '_limited.mat'))["CtDataLimited"][0,0]
        la_sino = torch.tensor(dataLimited["sinogram"])
        assert la_sino.shape == (n_known_angles[level-1], HTC2022_GEOMETRY.projection_size)
        known_angles = torch.tensor(dataLimited["parameters"]["angles"][0,0])
        to_rotate.append(int(known_angles[0, 0] / 0.5)+180) #angle is offset by 90 degrees
        sinos.append(torch.concat([la_sino, torch.zeros((HTC2022_GEOMETRY.n_projections-n_known_angles[level-1], HTC2022_GEOMETRY.projection_size))]).to(DEVICE, dtype=DTYPE))

    known_angles = torch.zeros(720, dtype=bool, device=DEVICE)
    known_angles[:n_known_angles[level-1]] = 1
    return torch.stack(sinos), known_angles, to_rotate, torch.stack(phantoms)

#Data generation
def rotation_matrix(angle: float):
    if isinstance(angle, torch.Tensor):
        tangle = angle.clone()
    else:
        tangle = torch.tensor(angle)
    c, s = torch.cos(tangle), torch.sin(tangle) 
    return torch.tensor([
        [c, -s],
        [s, c]
   ], device=DEVICE, dtype=DTYPE)

def get_angles(x: torch.Tensor):
    res = torch.angle(x[...,0] + 1j*x[...,1])
    # while (res < 0).any() or (res >= 2*torch.pi).any(): #uncomment for angles between o and 2pi
    #     res[res < 0] += 2*torch.pi
    #     res[res >= 2*torch.pi] -= 2*torch.pi
    return res #in range -pi to pi

class ConvergenceError(Exception):
    ...

def newton(f, x0:Union[float,torch.Tensor], tol = 1e-3, max_iters = 300):
    "return argmin(max) f, min(max) f"
    x = torch.tensor(x0, device=DEVICE, dtype=DTYPE)
    x.requires_grad_()
    xlast = None
    it = 0
    while xlast is None or torch.linalg.norm(x-xlast) >= tol:
        xlast = x.detach()
        y: torch.Tensor = f(x)
        y.backward(retain_graph=True)
        der: torch.Tensor = x.grad
        H: torch.Tensor = hessian(f, x)
        assert isinstance(H, torch.Tensor)
        h = -der / H
        x = xlast + h
        x.requires_grad_()
        it += 1
        if it > max_iters:
            raise ConvergenceError("Max iteration exceeded in Newton's method. Iteration is not converging (fast enough).")

    return x, f(x)
        

def disc_phantom(xy_minmax: Tuple[float, float, float, float], disc_radius: float, shape: Tuple[int, int], expected_num_inner_ellipses: int = 10, min_ellips_ratio = 0.25, max_ellips_ratio = 1.0):
    """Generate one phantom similar to the phantoms from the HTC dataset. A circle with value one with a number of radomly placed and tilted non-intersecting ellipses inside of it.

    Args:
        xy_minmax (Tuple[float, float, float, float]): [xmin, xmax, ymin, ymax] of reconstruction space. Can be chosen arbitrary as long as disc_radiues is appropriate scale.
        disc_radius (float): radius of cirlce at the center of the phantom.
        shape (Tuple[int, int]): shape of the phantom tensor to be generated.
        expected_num_inner_ellipses (int, optional): This regulates how densely the ellipses are placed. Defaults to 10.
        min_ellips_ratio (float, optional): ratio between the smallest allowed value and the maximum radius that can fit without intersection for ech randomly generated inner ellips. Must be < max_ellips_ratio. Defaults to 0.25.
        max_ellips_ratio (float, optional): ratio between the smallest allowed value and the maximum radius that can fit without intersection for ech randomly generated inner ellips. Must be <= 1.0. Defaults to 1.0.

    Returns:
        Tensor: a phantom consisting of the values 1 and 0. 
    """
    mx, Mx, my, My = xy_minmax
    NX, NY = shape
    assert Mx-mx > 2*disc_radius and My - my > 2*disc_radius, f"{disc_radius} radius is too large"
    assert max_ellips_ratio <= 1.0
    assert min_ellips_ratio < max_ellips_ratio
    res = torch.zeros(shape, device=DEVICE, dtype=DTYPE)
    margin_x, margin_y = (Mx-mx - 2*disc_radius)/2 * 0.8, (My-my-2*disc_radius)/2 * 0.8
    centerx = np.random.uniform((Mx+mx)/2-margin_x, (Mx+mx)/2+margin_x)
    centery = np.random.uniform((My+my)/2 - margin_y, (My+my)/2+margin_y)
    center = torch.tensor([centerx, centery], device=DEVICE, dtype=DTYPE)
    coords2D = torch.flip(torch.cartesian_prod(no_bdry_linspace(my, My, NY), no_bdry_linspace(mx, Mx, NX)).reshape(NY, NX, 2), dims=(0,-1))
    
    res[((coords2D - center)**2).sum(dim=-1) < disc_radius**2] = 1
    if expected_num_inner_ellipses == 0:
        return res

    def get_angle_span(ellips2disc: torch.Tensor, ri: float)->Tuple[torch.Tensor, torch.Tensor]:
        "angle span of an ellips"
        dummy_phii = 0
        dummy_centeri = torch.tensor([centerx+np.cos(dummy_phii)*ri,centery+np.sin(dummy_phii)*ri], device=DEVICE, dtype=DTYPE)
        def f(theta: torch.Tensor):
            return get_angles(ellips2disc @ torch.stack([torch.cos(theta), torch.sin(theta)]) + dummy_centeri - center) # between -pi and pi
        n_vertices = 5
        opts = []
        for i in range(n_vertices):
            thetai, anglei = newton(f, 2*torch.pi*i/n_vertices)
            opts.append((anglei, thetai))
        
        max_angle, theta2 = max(*opts)
        min_angle, theta1 = min(*opts)
        if max_angle-min_angle < 0.03:
            raise ConvergenceError("Same extrema found multiple times")

        return min_angle, max_angle

    start_phi = np.random.uniform(0,2*np.pi)
    angle = start_phi
    while angle - start_phi < 2*np.pi:
        max_ri = disc_radius*0.8
        ri = np.random.triangular(0.1*max_ri, max_ri, max_ri) #distance between ellips center and center of disc
        bound_segment = disc_radius*0.95
        
        max_ra = min(bound_segment-ri, ri) * max_ellips_ratio
        min_ra = min(bound_segment-ri, ri) * min_ellips_ratio
        ra = np.random.triangular(min_ra, (min_ra+max_ra)/2,  max_ra) #radius along main axis
        rb = ra * np.random.uniform(0.5, 1.0) #radius along second axis
        rel_tilt = np.random.uniform(0, 2*np.pi) #rotation of ellips

        Rrel = rotation_matrix(rel_tilt)
        disc2ellips = torch.tensor([ #change of coordinates from disc to ellips
            [1/ra, 0],
            [0, 1 / rb]
        ], device=DEVICE, dtype=DTYPE) @ Rrel

        ellips2disc = torch.linalg.inv(disc2ellips)
        try:
            ma, Ma = get_angle_span(ellips2disc, ri) #min_angle, max_angle
        except(ConvergenceError) as err:
            logging.debug("convergence failed:", err)
            # print("convergence failed:", err)
            continue
        
        phii = angle + np.random.exponential(2*np.pi/expected_num_inner_ellipses) - ma #angle to center of next ellips -- intent is to replicate a poisson process
        if phii + Ma + 0.1 > start_phi + 2*torch.pi: #risk of colliding with first ellips
            break
        disc2ellips = disc2ellips @ rotation_matrix(-phii)
        mat = disc2ellips.T@disc2ellips
        centerxi, centeryi = centerx + ri*torch.cos(phii), centery + ri*torch.sin(phii)
        centeri = torch.stack([centerxi, centeryi])
        res[torch.einsum("ijc,ck,ijk ->ij", coords2D-centeri, mat, coords2D-centeri) < 1] = 0
        angle = phii + Ma + 0.1
        ##DEBUG
        # disp = res + 0
        # zero_centered_phii = phii + 0
        # while zero_centered_phii > torch.pi:
        #     zero_centered_phii -= 2*torch.pi
        # disp[(get_angles(coords2D-center) >= zero_centered_phii+ma-0.05) & (get_angles(coords2D-center) < zero_centered_phii+ma+0.05)] = 2
        # disp[(get_angles(coords2D-center) >= zero_centered_phii+Ma-0.05) & (get_angles(coords2D-center) < zero_centered_phii+Ma+0.05)] = 2
        # plt.imshow(disp.cpu())
        # plt.figure()
        ##DEBUG

    return res

def disc_phantom_rects(xy_minmax: Tuple[float, float, float, float], disc_radius: float, shape: Tuple[int, int], expected_num_inner_squares: int = 10, min_side_ratio = 0.25, max_side_ratio = 1.0):
    """Generate one phantom similar to the phantoms from the HTC dataset. A circle with value one with a number of randomly placed and tilted non-intersecting rectangles inside of it.

    Args:
        xy_minmax (Tuple[float, float, float, float]): [xmin, xmax, ymin, ymax] of reconstruction space. Can be chosen arbitrary as long as disc_radiues is appropriate scale.
        disc_radius (float): radius of cirlce at the center of the phantom.
        shape (Tuple[int, int]): shape of the phantom tensor to be generated.
        expected_num_inner_ellipses (int, optional): This regulates how densely the ellipses are placed. Defaults to 10.
        min_ellips_ratio (float, optional): ratio between the smallest allowed value and the maximum radius that can fit without intersection for ech randomly generated inner ellips. Must be < max_ellips_ratio. Defaults to 0.25.
        max_ellips_ratio (float, optional): ratio between the smallest allowed value and the maximum radius that can fit without intersection for ech randomly generated inner ellips. Must be <= 1.0. Defaults to 1.0.

    Returns:
        Tensor: a phantom consisting of the values 1 and 0. 
    """
    mx, Mx, my, My = xy_minmax
    NX, NY = shape
    assert Mx-mx > 2*disc_radius and My - my > 2*disc_radius, f"{disc_radius} radius is too large"
    assert max_side_ratio <= 1.0
    assert min_side_ratio < max_side_ratio
    res = torch.zeros(shape, device=DEVICE, dtype=DTYPE)
    margin_x, margin_y = (Mx-mx - 2*disc_radius)/2 * 0.8, (My-my-2*disc_radius)/2 * 0.8
    centerx = np.random.uniform((Mx+mx)/2-margin_x, (Mx+mx)/2+margin_x)
    centery = np.random.uniform((My+my)/2 - margin_y, (My+my)/2+margin_y)
    center = torch.tensor([centerx, centery], device=DEVICE, dtype=DTYPE)
    coords2D = torch.flip(torch.cartesian_prod(no_bdry_linspace(my, My, NY), no_bdry_linspace(mx, Mx, NX)).reshape(NY, NX, 2), dims=(0,-1)) #xy format
    
    res[((coords2D - center)**2).sum(dim=-1) < disc_radius**2] = 1
    if expected_num_inner_squares == 0:
        return res

    def get_angle_span(square2disc: torch.Tensor, ri: float)->Tuple[torch.Tensor, torch.Tensor]:
        "angle span of a rectangle"
        square_center = torch.tensor([centerx+ri,centery], device=DEVICE, dtype=DTYPE) #fictive center
        corners_square = torch.tensor([
            [1,1], [-1,1], [-1,-1], [1,-1.0]
        ], device=DEVICE, dtype=DTYPE).reshape(4,2,1)
        corners_disc = (square2disc @ corners_square).reshape(4,2) + square_center
        opts = get_angles(corners_disc - center)
        min_angle, max_angle = opts.min(), opts.max()
        assert max_angle-min_angle>0.05
        return min_angle, max_angle

    start_phi = np.random.uniform(0,2*np.pi)
    angle = start_phi
    while angle - start_phi < 2*np.pi:
        max_ri = disc_radius*0.8
        ri = np.random.triangular(0.1*max_ri, max_ri, max_ri) #distance between square center and center of disc
        bound_segment = disc_radius*0.95
        
        max_sa = min(bound_segment-ri, ri)/(2**0.5) * max_side_ratio #max distance from rectangle center is sqrt(2)sa
        min_sa = min(bound_segment-ri, ri)/(2**0.5) * min_side_ratio
        sa = np.random.triangular(min_sa, (min_sa+max_sa)/2,  max_sa) #long side of rectangle is 2sa
        sb = sa * np.random.uniform(0.5, 1.0) #short side of rectangle is 2sb
        rel_tilt = np.random.uniform(0, 2*np.pi) #rotation of rectangle

        Rrel = rotation_matrix(rel_tilt)
        disc2square = torch.tensor([ #change of coordinates from disc to square
            [1/sa, 0],
            [0, 1 / sb]
        ], device=DEVICE, dtype=DTYPE) @ Rrel

        square2disc = torch.linalg.inv(disc2square)
        ma, Ma = get_angle_span(square2disc, ri) #min_angle, max_angle
        
        phii = angle + np.random.exponential(2*np.pi/expected_num_inner_squares) - ma #angle to center of next ellips -- intent is to replicate a poisson process
        if phii + Ma + 0.1 > start_phi + 2*torch.pi: #risk of colliding with first rectangle
            break
        disc2square = disc2square @ rotation_matrix(-phii)
        centerxi, centeryi = centerx + ri*torch.cos(phii), centery + ri*torch.sin(phii)
        centeri = torch.stack([centerxi, centeryi])
        res[torch.einsum("ck,ijk->ijc", disc2square, coords2D-centeri).abs().max(dim=-1).values<1] = 0
        angle = phii + Ma + 0.1
        ##DEBUG
        # disp = res + 0
        # zero_centered_phii = phii + 0
        # while zero_centered_phii > torch.pi:
        #     zero_centered_phii -= 2*torch.pi
        # disp[(get_angles(coords2D-center) >= zero_centered_phii+ma-0.05) & (get_angles(coords2D-center) < zero_centered_phii+ma+0.05)] = 2
        # disp[(get_angles(coords2D-center) >= zero_centered_phii+Ma-0.05) & (get_angles(coords2D-center) < zero_centered_phii+Ma+0.05)] = 2
        # plt.figure()
        # plt.imshow(disp.cpu())
        # plt.title(f"{zero_centered_phii:.2}+{Ma:.2},{ma:.2}")
        # plt.colorbar()
        ##DEBUG

    return res


if __name__ == '__main__':
    import matplotlib
    # matplotlib.use("WebAgg")
    import matplotlib.pyplot as plt

    print("hellu")
    xy_minmax = [-38, 38, -38, 38.0]
    disc_radius = 35.0

    # for _ in range(5):
    #     res = disc_phantom_rects(xy_minmax, disc_radius, (512, 512), 20, 0.8, 0.9)
    #     plt.figure()
    #     plt.imshow(res.cpu())
    #     plt.colorbar()
    for i in range(5):
        res = disc_phantom(xy_minmax, disc_radius, (512, 512), 20, 0.8, 0.9)
        plt.figure()
        plt.imshow(res.cpu(), cmap="gray")
        plt.gca().set_axis_off()
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(f"data{i}.png", bbox_inches = 'tight',
            pad_inches = 0)
    # for _ in range(5):
    #     res = disc_phantom_rects(xy_minmax, disc_radius, (512, 512), 30, 0.4, 0.6)
    #     plt.figure()
    #     plt.imshow(res.cpu())
    #     plt.colorbar()
    

    plt.show()


        