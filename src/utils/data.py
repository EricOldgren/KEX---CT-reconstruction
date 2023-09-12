import torch
import numpy as np
import scipy
from typing import Tuple, Union

from torch.autograd.functional import hessian

from utils.tools import DEVICE, DTYPE, no_bdry_linspace, GIT_ROOT

#Data loading
def get_htc2022_train_phantoms():
    return torch.stack(torch.load( GIT_ROOT / "data/HTC2022/HTCTrainingPhantoms.pt", map_location=DEVICE)).to(DTYPE)


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

def optimize(f,x0: torch.Tensor, maximize = False, max_iters = 500, tol=1e-4):
    "return argmin(max) f, min(max) f"
    x = x0.clone()
    x_last = None
    x.requires_grad_()
    optimizer = torch.optim.SGD([x], lr=0.001)
    for it in range(max_iters):
        res: torch.Tensor = f(x)
        print("\tIter:", it, "res:", res)
        if maximize:
            res*=-1
        res.backward()
        # if x_last is not None and torch.linalg.norm(x-x_last) < tol:
        #     break
        x_last = x.detach()
        optimizer.step()

    return x, f(x)

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
    start_phi = np.random.uniform(0,2*np.pi)
    def get_angles(x: torch.Tensor):
        res = torch.angle(x[...,0] + 1j*x[...,1])
        # while (res < 0).any() or (res >= 2*torch.pi).any():
        #     res[res < 0] += 2*torch.pi
        #     res[res >= 2*torch.pi] -= 2*torch.pi
        return res
    def get_angle_span(ellips2disc: torch.Tensor, ri: float)->Tuple[torch.Tensor, torch.Tensor]:
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

        ##
        ##DEBUGGING
        # print("max angle:", max_angle, "min_angle:", min_angle)
        # print("thetas:", theta1, theta2)
        # edge_point1 = ellips2disc @ torch.stack([torch.cos(theta1), torch.sin(theta1)]) + dummy_centeri
        # edge_inds1 = xy2inds(edge_point1)
        # edge_point2 = ellips2disc @ torch.stack([torch.cos(theta2), torch.sin(theta2)]) + dummy_centeri
        # edge_inds2 = xy2inds(edge_point2)
        # disc2ellips = torch.linalg.inv(ellips2disc)
        # print("edge points:", edge_point1, edge_point2)
        # print("edge angles:", get_angles(edge_point1-center), get_angles(edge_point2-center))
        # mat = disc2ellips.T@disc2ellips
        # disp = torch.zeros(shape, device=DEVICE, dtype=DTYPE)
        # disp[((coords2D - center)**2).sum(dim=-1) < disc_radius**2] = 1
        # disp[torch.einsum("ijc,ck,ijk ->ij", coords2D-dummy_centeri, mat, coords2D-dummy_centeri) < 1] = 0
        # disp[(get_angles(coords2D-center) < (max_angle + 0.1)) & (get_angles(coords2D-center) >= (max_angle - 0.1)) ] = 0.5
        # disp[edge_inds1[1]-5:edge_inds1[1]+5, edge_inds1[0]-5:edge_inds1[0]+5] = 2
        # disp[edge_inds2[1]-5:edge_inds2[1]+5, edge_inds2[0]-5:edge_inds2[0]+5] = 2
        # plt.imshow(disp.cpu())
        # plt.colorbar()
        # plt.figure()
        ###DEBUGGING

        return min_angle, max_angle

    angle = start_phi
    ##DEBUGGING
    # angles_img = get_angles(coords2D - center)
    # plt.imshow(angles_img.cpu())
    # plt.title("angles")
    # plt.colorbar()
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(coords2D[...,0].cpu())
    # plt.subplot(122)
    # plt.imshow(coords2D[...,1].cpu())
    # plt.title("cords")
    # plt.colorbar()
    # plt.figure()
    # print("xy_minmax:", xy_minmax)
    # print("start angle:", start_phi)
    ##DEBUGGING


    while angle - start_phi < 2*np.pi:
        max_ri = disc_radius*0.8
        ri = np.random.triangular(0.1*max_ri, max_ri, max_ri)
        
        max_ra = min(max_ri-ri, ri) * max_ellips_ratio
        min_ra = min(max_ri-ri, ri) * min_ellips_ratio
        ra = np.random.triangular(min_ra, (min_ra+max_ra)/2,  max_ra)
        rb = ra / np.random.uniform(0.5, min(2, max_ri / ra))
        rel_tilt = np.random.uniform(0, 2*np.pi)

        Rrel = rotation_matrix(rel_tilt)
        disc2ellips = torch.tensor([
            [1/ra, 0],
            [0, 1 / rb]
        ], device=DEVICE, dtype=DTYPE) @ Rrel

        ellips2disc = torch.linalg.inv(disc2ellips)
        try:
            ma, Ma = get_angle_span(ellips2disc, ri)
        except(ConvergenceError) as err:
            print("convergence failed:", err)
            continue
        
        phii = angle + np.random.exponential(2*np.pi/expected_num_inner_ellipses) - ma #angle to center of ellips
        if phii + Ma + 0.1 > start_phi + 2*torch.pi:
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

    #DEBUG
    # plt.imshow(res.cpu())
    # plt.title("res")
    # plt.colorbar()
    # plt.figure()
    #DEBUG
    return res





if __name__ == '__main__':
    import matplotlib
    matplotlib.use("WebAgg")
    import matplotlib.pyplot as plt
    xy_minmax = [-1,1,-1,1]
    disc_radius = 0.8
    phantom_shape = (512, 512)


    generated_synthetic_data = []
    scales_counts = [
        (1.0, 0.9, 3), #max_ellips_ratio, min_ellips_ratio, n_phantoms
        (1.0, 0.8, 3),
        (1.0, 0.7, 3),
        (0.9, 0.5, 3),
        (0.9, 0.3, 3),
        (0.5, 0.2, 3),
        (0.5, 0.1, 3)
    ]
    density_levels = [0, 10, 15, 20]

    for (M, m, N) in scales_counts:
        for n_phantoms in density_levels:
            for i in range(N):
                phantom = better_disc_phantom(xy_minmax, disc_radius, phantom_shape, n_phantoms, m, M)
                if i == 0:
                    plt.figure()
                    plt.imshow(phantom.cpu())
                    plt.colorbar()
                    plt.title(f"M,m,n_ph:{M},{m},{n_phantoms}")
                generated_synthetic_data.append(phantom)
    
    plt.show()

    generated_synthetic_data = torch.stack(generated_synthetic_data)
    save_path = GIT_ROOT / "data/synthetic_htc_data.pt"
    torch.save(generated_synthetic_data, save_path)
    print("Data saved to:", save_path)



        