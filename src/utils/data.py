import torch
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Union

from torch.autograd.functional import hessian

from utils.tools import DEVICE, DTYPE, no_bdry_linspace, GIT_ROOT

#Data loading
def get_htc2022_train_phantoms():
    return torch.stack(torch.load( GIT_ROOT / "data/HTC2022/HTCTrainingPhantoms.pt", map_location=DEVICE)).to(DTYPE)
def get_synthetic_htc_phantoms():
    "retrieve generated phantoms designed to look like the HTC data phantoms"
    return torch.concat([
        torch.load(GIT_ROOT / "data/synthetic_htc_data.pt", map_location=DEVICE),
        torch.load(GIT_ROOT / "data/synthetic_htc_harder_data.pt", map_location=DEVICE)
    ])
def get_htc_trainval_phantoms():
    """retrieve train_phantoms, validation_phantoms
        return train_set, validation_phantoms
    """
    return get_synthetic_htc_phantoms(), get_htc2022_train_phantoms()
    # return train_test_split(get_synthetic_htc_phantoms(), test_size=0.2)
    # htc_phantoms = get_htc2022_train_phantoms()
    # train_phantoms, validation_phantoms = train_test_split(htc_phantoms, test_size=3)
    # train_set = torch.concat([train_phantoms, get_synthetic_htc_phantoms()])
    # return train_set, validation_phantoms


def get_kits_train_phantoms():
    return torch.load(GIT_ROOT / "data/kits_phantoms_256.pt", map_location=DEVICE)[:500, 1]
def get_kits_test_phantom():
    return torch.load(GIT_ROOT / "data/kits_phantoms_256.pt", map_location=DEVICE)[500:, 1]

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

def random_disc_phantom(xy_minmax: Tuple[float, float, float, float], disc_radius: float, shape: Tuple[int, int], n_inner_ellipses: int = 10):
    mx, Mx, my, My = xy_minmax
    NX, NY = shape
    assert Mx-mx > 2*disc_radius and My - my > 2*disc_radius, f"{disc_radius} radius is too large"
    res = torch.zeros(shape, device=DEVICE, dtype=DTYPE)
    margin_x, margin_y = (Mx-mx - 2*disc_radius)/2 * 0.8, (My-my-2*disc_radius)/2 * 0.8
    centerx = np.random.uniform((Mx+mx)/2-margin_x, (Mx+mx)/2+margin_x)
    centery = np.random.uniform((My+my)/2 - margin_y, (My+my)/2+margin_y)
    center = torch.tensor([centery, centerx], device=DEVICE, dtype=DTYPE)
    coords2D = torch.cartesian_prod(no_bdry_linspace(my, My, NY), no_bdry_linspace(mx, Mx, NX)).reshape(NY, NX, 2)
    res[((coords2D - center)**2).sum(dim=-1) < disc_radius**2] = 1

    for _ in range(n_inner_ellipses):
        max_ri = disc_radius*0.95
        ri = np.random.triangular(0, max_ri, max_ri)
        phii = np.random.uniform(0, 2*np.pi)
        centerxi, centeryi = centerx + ri*np.cos(phii), centery + ri*np.sin(phii)
        ceneteri = torch.tensor([centeryi, centerxi], device=DEVICE, dtype=DTYPE)

        ra = np.random.triangular(0, (max_ri-ri)/4,  (max_ri-ri)/2)
        rb = ra / np.random.uniform(0.5, min(2, max_ri / ra))
        tilt = np.random.uniform(0, 2*np.pi)

        mat = rotation_matrix(tilt) @ torch.tensor([
            [1/ra**2, 0],
            [0, 1 / rb**2]
        ], device=DEVICE, dtype=DTYPE) @ rotation_matrix(tilt).T
        res[torch.einsum("ijc,ck,ijk ->ij", coords2D-ceneteri, mat, coords2D-ceneteri) < 1] = 0
        # res[((coords2D - ceneteri)**2).sum(dim=-1) < inner_disc_radius_i**2] = 0

    return res

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
        

def better_disc_phantom(xy_minmax: Tuple[float, float, float, float], disc_radius: float, shape: Tuple[int, int], expected_num_inner_ellipses: int = 10, min_ellips_ratio = 0.25, max_ellips_ratio = 1.0):
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

    def get_angles(x: torch.Tensor):
        res = torch.angle(x[...,0] + 1j*x[...,1])
        # while (res < 0).any() or (res >= 2*torch.pi).any(): #uncomment for angles between o and 2pi
        #     res[res < 0] += 2*torch.pi
        #     res[res >= 2*torch.pi] -= 2*torch.pi
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
            print("convergence failed:", err)
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


if __name__ == '__main__':
    print("this is the data file!!!!")
    import matplotlib
    matplotlib.use("WebAgg")
    import matplotlib.pyplot as plt
    xy_minmax = [-1,1,-1,1]
    disc_radius = 0.8
    phantom_shape = (512, 512)

    generated_synthetic_data = []
    settings = [
        (5, 0.9, 1.0, 50), #n_ellipses, min_ellips_ratio, max_ellips_ratio, n_phantoms
        (10, 0.9, 1.0, 50),
        (20, 0.9, 1.0, 50),
        (5, 0.8, 1.0, 50),
        (10, 0.8, 1.0, 50),
        (20, 0.8, 1.0, 50),
        (30, 0.5, 0.6, 50),
        (50, 0.5, 0.6, 50),
        (40, 0.4, 0.5, 50),
        (60, 0.4, 0.5, 50)
    ]

    for n_ellipses, m, M, N in settings:
            for i in range(N):
                phantom = better_disc_phantom(xy_minmax, disc_radius, phantom_shape, n_ellipses, m, M)
                if i < 2:
                    plt.figure()
                    plt.imshow(phantom.cpu())
                    plt.colorbar()
                    plt.title(f"M,m,n_ph:{M},{m},{n_ellipses}")
                generated_synthetic_data.append(phantom)
            print("Data generated with settings:", n_ellipses, m, M, N)
    
    plt.show()

    generated_synthetic_data = torch.stack(generated_synthetic_data)
    save_path = GIT_ROOT / "data/synthetic_htc_harder_data.pt"
    torch.save(generated_synthetic_data, save_path)
    print("Data saved to:", save_path)



        