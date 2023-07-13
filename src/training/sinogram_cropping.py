import torch
import numpy as np
import random
from utils.geometry import ParallelGeometry

FLOAT_EPS = np.finfo(float).eps

def crop_sinos(X: torch.Tensor, ar: float, full_geometry: ParallelGeometry)->"tuple[torch.Tensor, tuple[float, float]]":
    """Given tensors representing sinograms sampled at angles [0, pi] crops these at a randomly chosen start angle and returns the cropped sinograms together with the angles they are known at.

    Args:
        X (torch.Tensor) of shape N x phi_size x t_size: batch of sinograms
        ar (float): angle ratio to crop sinograms to
        
    returns X_limited, (start_angle, end_angle) -- sinos known at [start_angle, end_angle) angles are taken from geometry angles
    """
    assert 0 < ar <= 1.0
    assert full_geometry.ar == 1.0
    N, Np, Nt = X.shape
    Np_limited = int(ar*Np)
    
    start_ind = random.randint(0, Np-1)
    start_angle = full_geometry.angles[start_ind]
    n_in_range = Np - start_ind
    n_wrapped = Np_limited - n_in_range
    if n_wrapped > 0:
        X_limited = torch.concat([
            X[:, start_ind:], torch.flip(X[:, :n_wrapped], dims=(-1,))
        ], dim=1)
        return X_limited, (start_angle, np.pi + full_geometry.angles[n_wrapped])
    
    return X[:, start_ind:start_ind+Np_limited], (start_angle, full_geometry.angles[start_ind+Np_limited])

def zeropad_limited_angle_sinos(X: torch.Tensor, full_geometry: ParallelGeometry):
    assert full_geometry.ar == 1.0
    N, lim_Np, Nt = X.shape
    Np = full_geometry.phi_size
    return torch.concat([X, torch.zeros((N, Np-lim_Np,Nt), dtype=X.dtype, device=X.device)], dim=1)
    
    
def rotate_sinos(X: torch.Tensor, angle_interval: "tuple[float, float]", desired_angle_interval: "tuple[float, float]" = (0,np.pi)):
    """Move the rows in a batch of sinograms uniformly sampled at given angle interval to obtain the sinograms sampled at the desired angle interval.
    Both intervals must be of length pi. This corresponds to rotating the phantom.
       
    Args:
        X (torch.Tensor) of shape N x phi_size x t_size: batch of siograms
        angle_interval (tuple[float, float]): (start_angle, end_angle) interval sinograms are sampled in
        desired_angle_interval (tuple[float, float], optional):(start_angle, end_angle) interval sinograms should be converted to. Defaults to (0,np.pi).
        
    returns: X_rotated
    """
    N, Np, Nt = X.shape
    assert abs(angle_interval[1]-angle_interval[0] - np.pi) < 10*FLOAT_EPS and abs(desired_angle_interval[1]-desired_angle_interval[0] - np.pi) < 10*FLOAT_EPS, "length of both angle intervals must be pi!"
    (start_angle, end_angle), (desired_start_angle, desired_end_angle) = angle_interval, desired_angle_interval
    assert 0 <= start_angle < 2*np.pi and 0 < end_angle <= 2*np.pi and 0<= desired_start_angle < 2*np.pi and 0 < desired_end_angle <= 2*np.pi
    
    if start_angle < desired_start_angle : #shifting positive amount
        n_overlapped = int((end_angle - desired_start_angle) / np.pi * Np)
        n_wrapped = Np - n_overlapped
        return torch.concat([
            X[:, n_wrapped:], torch.flip(X[:, :n_wrapped], dims=(-1,))
        ], dim=1)
    else:
        n_overlapped = int((desired_end_angle - start_angle) / np.pi * Np)
        n_wrapped = Np - n_overlapped
        return torch.concat([
            torch.flip(X[:, -n_wrapped:], dims=(-1,)), X[:, :-n_wrapped]
        ], dim=1)
        

if __name__ == "__main__":
    
    from models.analyticmodels import RamLak
    from utils.geometry import ParallelGeometry, DEVICE
    import odl.contrib.torch as odl_torch
    import matplotlib.pyplot as plt
    from utils.inverse_moment_transform import extrapolate_sinos
    g = ParallelGeometry(1.0, 400, 400)
    ray_l = odl_torch.OperatorModule(g.ray)
    
    ramlak = RamLak(g)
    phantom = torch.zeros((256, 256), dtype=float, device=DEVICE)[None]
    phantom[:, 100:156, 100:156] = 1.0
    sinos = ray_l(phantom)
    rotated_sinos = rotate_sinos(sinos, (np.pi/4, 5*np.pi/4)) #45 deg rotation
    cropped_sinos, angle_interval = crop_sinos(sinos, 0.5, g)
    limited_angle_sinos = zeropad_limited_angle_sinos(cropped_sinos, g)
    limited_angle_sinos = rotate_sinos(limited_angle_sinos, (angle_interval[0], angle_interval[0]+np.pi)) #move sinogram to region (0,pi)
    estimated_data = extrapolate_sinos(cropped_sinos, torch.linspace(angle_interval[0], angle_interval[1], cropped_sinos.shape[1]), torch.linspace(angle_interval[1], angle_interval[0]+torch.pi, g.phi_size-cropped_sinos.shape[1]), N_moments=500)
    exp_sinos = torch.concat([cropped_sinos, estimated_data], dim=1)
    exp_sinos = rotate_sinos(exp_sinos, (angle_interval[0], angle_interval[0]+np.pi))
    
    fig, _ = plt.subplots(1,2)
    fig.suptitle("full sinos")
    plt.subplot(121)
    plt.imshow(sinos[0].cpu())
    plt.title("sinos")
    plt.subplot(122)
    plt.imshow(ramlak(sinos)[0].cpu())
    plt.title("recon")
    fig.show()
    
    fig1, _ = plt.subplots(1, 2)
    fig1.suptitle("rotated sinos")
    plt.subplot(121)
    plt.imshow(rotated_sinos[0].cpu())
    plt.title("rotated sino")
    plt.subplot(122)
    plt.imshow(ramlak(rotated_sinos)[0].cpu())
    plt.title("recon")
    fig1.show()
    
    fig2, _ = plt.subplots(1,2)
    fig2.suptitle(f"cropped sinograms")
    plt.subplot(121)
    plt.imshow(limited_angle_sinos[0].cpu())
    plt.title("sino")
    plt.subplot(122)
    plt.imshow(ramlak(limited_angle_sinos)[0].cpu())
    plt.title("recon")
    fig2.show()
    
    fig3, _ = plt.subplots(1, 2)
    fig3.suptitle("extrapolated sinos")
    plt.subplot(121)
    plt.imshow(exp_sinos[0].cpu())
    plt.title("rotated sino")
    plt.subplot(122)
    plt.imshow(ramlak(exp_sinos)[0].cpu())
    plt.title("recon")
    
    plt.show()    