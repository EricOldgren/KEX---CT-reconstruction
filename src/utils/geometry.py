import odl
import odl.contrib.torch as odl_torch
import torch
import numpy as np
from .data_generator import unstructured_random_phantom, random_phantom
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BackProjection(odl.Operator):

    def __init__(self, reco_space, geometry):

        self.ray = odl.tomo.RayTransform(reco_space, geometry)
        super().__init__(self.ray.range, self.ray.domain, linear=True)
    
    def _call(self, x):
        return self.ray.adjoint(x)

    @property
    def adjoint(self):
        return self.ray

class BasicModel(nn.Module):

    def __init__(self, angle_ratio, phi_size, t_size, kernel = None, **kwargs):
        super(BasicModel, self).__init__(**kwargs)
        # Reconstruction space: discretized functions on the rectangle [-20, 20]^2 with 300 samples per dimension.
        reco_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20], shape=[256, 256], dtype='float32')
        # Angles: uniformly spaced, n = 1000, min = 0, max = pi
        angle_partition = odl.uniform_partition(0, np.pi*angle_ratio, phi_size)
        # Detector: uniformly sampled, n = 500, min = -30, max = 30
        detector_partition = odl.uniform_partition(-30, 30, t_size)

        # Make a parallel beam geometry with flat detector
        geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

        BP = BackProjection(reco_space, geometry)
        self.BP_layer = odl_torch.OperatorModule(BP)

        if kernel == None:
            self.kernel = torch.randn((int(np.ceil(0.5 + 0.5*t_size)),)).to(DEVICE)
        else:
            self.kernel = kernel
        self.kernel.requires_grad_(True)
    
    def forward(self, sinos):
        filtered_sinos = torch.fft.irfft(torch.fft.rfft(sinos)*self.kernel)

        return self.BP_layer(filtered_sinos)
    



def setup(angle_ratio = 1.0, phi_size = 100, t_size = 300, num_samples = 1000, train_ratio=0.8):
    """
        creates back projection layer and generates random data in the given angle ratio
        parameters
            :angle_ratio - angle gemoetry is 0 to pi * angle_ratio
            :phi_size - number of angles for measurements / ray transform
            :t_size - number of lines per angle 

        return  (train_sinos, train_y, test_sinos, test_y), BP_layer
    """

    # full_data: torch.Tensor = torch.load("/content/kits_phantoms_256.pt").moveaxis(0,1)
    # full_data = torch.concat([full_data[1], full_data[0], full_data[2]])
    # full_data = full_data[:600]
    # Reconstruction space: discretized functions on the rectangle [-20, 20]^2 with 300 samples per dimension.
    reco_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20], shape=[256, 256], dtype='float32')
    # Angles: uniformly spaced, n = 1000, min = 0, max = pi
    angle_partition = odl.uniform_partition(0, np.pi*angle_ratio, phi_size)
    # Detector: uniformly sampled, n = 500, min = -30, max = 30
    detector_partition = odl.uniform_partition(-30, 30, t_size)

    # Make a parallel beam geometry with flat detector
    geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

    ray = odl.tomo.RayTransform(reco_space, geometry)
    ray_layer = odl_torch.OperatorModule(ray)

    BP = BackProjection(reco_space, geometry)
    BP_layer = odl_torch.OperatorModule(BP)


    full_data = torch.tensor([])
    additional_data = random_phantom(reco_space=reco_space,num_ellipses=7).asarray()[None]
    for i in range(num_samples):
        additional_data = np.concatenate([additional_data, unstructured_random_phantom(reco_space=reco_space,num_ellipses=7).asarray()[None]])

    full_data=torch.concat((full_data,torch.from_numpy(additional_data) )).to(DEVICE)

    print("Calculating sinograms...")
    sinos: torch.Tensor = ray_layer(full_data)

    n_training = int(num_samples*train_ratio)
    train_y, train_sinos = full_data[:n_training], sinos[:n_training]
    test_y, test_sinos = full_data[n_training:], sinos[n_training:]

    print("Training data shape ", train_y.shape)

    return (train_sinos, train_y, test_sinos, test_y), BP_layer