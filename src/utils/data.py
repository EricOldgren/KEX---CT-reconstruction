import torch
import numpy as np
from typing import Tuple

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

def generate_htclike_batch(n_phantoms: int = 10, n_inner_ellipses = 5):
    res = torch.zeros((n_phantoms, 512, 512), device=DEVICE, dtype=DTYPE)
    for i in range(n_phantoms):
        res[i] = random_disc_phantom([-1,1,-1,1], 0.9, [512, 512], n_inner_ellipses)

    return res
def get_htclike_train_phantoms():
    res = torch.zeros((25, 512, 512), device=DEVICE, dtype=DTYPE)
    for i, n_ellipses in enumerate([0, 5, 10, 15, 20]):
        res[i*5:(i+1)*5] = generate_htclike_batch(5, n_ellipses)

    return res


if __name__ == '__main__':
    import matplotlib
    matplotlib.use("WebAgg")
    import matplotlib.pyplot as plt
    xy_minmax = [-1,1,-1,1]

    for _ in range(10):
        print("="*40)
        phantom = random_disc_phantom(xy_minmax, 0.8, [512, 512], 15)
        plt.figure()
        plt.imshow(phantom.cpu())
    
    plt.show()


        