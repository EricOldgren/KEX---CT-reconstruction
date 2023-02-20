import odl
import odl.contrib.torch as odl_torch
import torch
import numpy as np
from .data_generator import unstructured_random_phantom, random_phantom
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BackProjection(odl.Operator):

    def __init__(self, ray: odl.tomo.RayTransform):
        self.ray = ray
        super(BackProjection, self).__init__(self.ray.range, self.ray.domain, linear=True)

    @classmethod
    def from_space(clc, reco_space, geometry):
        return BackProjection(odl.tomo.RayTransform(reco_space, geometry))
    
    def _call(self, x):
        return self.ray.adjoint(x)

    @property
    def adjoint(self):
        return self.ray

class Geometry:
    
    def __init__(self, angle_ratio: float, phi_size: int, t_size: int):
        """
            Create a parallel beam geometry with corresponding forward and backward projections

            parameters
                :angle_ratio - angle gemoetry is 0 to pi * angle_ratio
                :phi_size - number of angles for measurements, i.e angle resolution
                :t_size - number of lines per angle, i.e detector resolution
        """
        self.ar = angle_ratio; self.phi_size = phi_size; self.t_size = t_size

        # Reconstruction space: discretized functions on the rectangle [-20, 20]^2 with 300 samples per dimension.
        self.reco_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20], shape=[256, 256], dtype='float32')
        # Angles: uniformly spaced, n = phi_size, min = 0, max = ratio*pi
        self.angle_partition = odl.uniform_partition(0, np.pi*angle_ratio, phi_size)
        # Detector: uniformly sampled, n = 500, min = -30, max = 30
        self.detector_partition = odl.uniform_partition(-30, 30, t_size)

        # Make a parallel beam geometry with flat detector
        self.geometry = odl.tomo.Parallel2dGeometry(self.angle_partition, self.detector_partition)

        self.ray = odl.tomo.RayTransform(self.reco_space, self.geometry)

        self.BP = BackProjection(self.ray)

class BasicModel(nn.Module):

    def __init__(self, geometry: Geometry, kernel: torch.Tensor = None, **kwargs):
        "Linear layer consisting of a 1D sinogram kernel in frequency domain"
        super(BasicModel, self).__init__(**kwargs)
        
        self.geometry = geometry
        self.BP_layer = odl_torch.OperatorModule(geometry.BP)

        if kernel == None:
            self.kernel = torch.randn((int(np.ceil(0.5 + 0.5*geometry.t_size)),)).to(DEVICE)
        else:
            assert kernel.shape == (int(np.ceil(0.5 + 0.5*geometry.t_size)),), f"wrong formatted specific kernel {kernel.shape} for geometry {geometry}"
            self.kernel = kernel.to(DEVICE)
        self.kernel.requires_grad_(True)
    
    def forward(self, sinos):
        filtered_sinos = torch.fft.irfft(torch.fft.rfft(sinos)*self.kernel)

        return self.BP_layer(filtered_sinos)
    

def setup(angle_ratio = 1.0, phi_size = 100, t_size = 300, num_samples = 1000, train_ratio=0.8):
    """
        Creates Geometry with appropriate forward and backward projections in the given angle ratio and generates random data as specified
        parameters
            :angle_ratio - angle gemoetry is 0 to pi * angle_ratio
            :phi_size - number of angles for measurements / ray transform
            :t_size - number of lines per angle
            :num_samples - number of randomly generated datapoints
            :train_ratio - ratio of generated samples used for training data

        return  (train_sinos, train_y, test_sinos, test_y), geometry
    """

    # read_data: torch.Tensor = torch.load("/data/kits_phantoms_256.pt").moveaxis(0,1)
    # read_data = torch.concat([read_data[1], read_data[0], read_data[2]])
    # read_data = read_data[:600] # -- uncomment to read this data
    read_data = torch.tensor([])

    geometry = Geometry(angle_ratio, phi_size, t_size)

    ray_layer = odl_torch.OperatorModule(geometry.ray)

    constructed_data = unstructured_random_phantom(reco_space=geometry.reco_space,  num_ellipses=10).asarray()[None]
    for _ in range(num_samples):
        constructed_data = np.concatenate([constructed_data, unstructured_random_phantom(reco_space=geometry.reco_space, num_ellipses=10).asarray()[None]])

    full_data=torch.concat((read_data,torch.from_numpy(constructed_data) )).to(DEVICE)

    print("Calculating sinograms...")
    sinos: torch.Tensor = ray_layer(full_data)

    n_training = int(num_samples*train_ratio)
    train_y, train_sinos = full_data[:n_training], sinos[:n_training]
    test_y, test_sinos = full_data[n_training:], sinos[n_training:]

    print("Constructed training dataset of shape ", train_y.shape)

    return (train_sinos, train_y, test_sinos, test_y), geometry