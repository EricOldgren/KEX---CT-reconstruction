import torch
import numpy as np
from odl import DiscretizedSpace
from odl.phantom import ellipsoid_phantom

from utils.tools import DEVICE, DTYPE, no_bdry_linspace

from typing import Tuple

def random_ellipsoid(min_pt, max_pt, value = 0.5):
    "Generate an ellips in the rectangular region bordered by mint_pt, max_pt"
    (mx, my), (Mx, My) = min_pt, max_pt
    dx, dy = Mx-mx, My-my
    centerx = np.random.uniform(mx+dx*0.05, Mx-dx*0.05)
    centery = np.random.uniform(my+dy*0.05, My-dy*0.05)
    Rx = np.min((Mx-centerx, centerx-mx))
    Ry = np.min((My-centery, centery-my))
    rx, ry = np.random.random()*Rx, np.random.random()*Ry
    #Ellipse
    #        value  axisx  axisy   x    y   rotation(rad)
    return [value, rx,     ry,centerx, centery, np.random.random()*np.pi]

def unstructured_random_phantom(reco_space: DiscretizedSpace, num_ellipses = 10):
    "Phantom of random ellipses. More random and doesn't look like a brain. Maybe useful for more diverse data."

    ellipsoids = []
    for _ in range(num_ellipses):
        ellipsoids.append(random_ellipsoid([-1.0, -1.0], [1.0, 1.0], value = np.random.uniform(0.1, 0.6)))

    res = ellipsoid_phantom(reco_space, ellipsoids)
    res /= np.max(res)

    return res

# def square_phantom(xy_minmax: Tuple[float, float, float, float], shape: Tuple[int, int], n_squares: int = 10, minmax_s: Tuple[float, float] = (0.2, 0.8)):
#     mx, Mx, my, My = xy_minmax
#     NX, NY = shape
#     D = min(Mx-mx, My-my)
#     mx, Ms = minmax_s
#     res = torch.zeros(shape, device=DEVICE, dtype=DTYPE)

#     coords2D = torch.cartesian_prod(no_bdry_linspace(my, My, NY), no_bdry_linspace(mx, Mx, NX)).reshape(NY, NX, 2)

#     for _ in range(n_squares):
        
#         s = np.random.uniform(ms, Ms)*D

#         centerxi, centeryi = np.random.normal() , centery + ri*np.sin(phii)
#         ceneteri = torch.tensor([centeryi, centerxi], device=DEVICE, dtype=DTYPE)

#         ra = np.random.triangular(0, (max_ri-ri)/4,  (max_ri-ri)/2)
#         rb = ra / np.random.uniform(0.5, min(2, max_ri / ra))
#         tilt = np.random.uniform(0, 2*np.pi)

#         mat = rotation_matrix(tilt) @ torch.tensor([
#             [1/ra**2, 0],
#             [0, 1 / rb**2]
#         ], device=DEVICE, dtype=DTYPE) @ rotation_matrix(tilt).T
#         res[torch.einsum("ijc,ck,ijk ->ij", coords2D-ceneteri, mat, coords2D-ceneteri) < 1] = 0
#         # res[((coords2D - ceneteri)**2).sum(dim=-1) < inner_disc_radius_i**2] = 0

#     return res

    