import odl
import odl.contrib.torch as odl_torch
import torch
import numpy as np
from .data_generator import unstructured_random_phantom, random_phantom
import torch.nn as nn
import random
import matplotlib.pyplot as plt

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
    
    def __init__(self, angle_ratio: float, phi_size: int, t_size: int, reco_shape = (256, 256)):
        """
            Create a parallel beam geometry with corresponding forward and backward projections

            parameters
                :angle_ratio - angle gemoetry is 0 to pi * angle_ratio
                :phi_size - number of angles for measurements, i.e angle resolution
                :t_size - number of lines per angle, i.e detector resolution
                :reco_shape - pixel shape of images to be reconstructed
        """
        self.ar = angle_ratio; self.phi_size = phi_size; self.t_size = t_size

        # Reconstruction space: discretized functions on the rectangle [-20, 20]^2 with 300 samples per dimension.
        self.reco_space = odl.uniform_discr(min_pt=[-1.0, -1.0], max_pt=[1.0, 1.0], shape=reco_shape, dtype='float32')
        self.rho = np.linalg.norm(self.reco_space.max_pt - self.reco_space.min_pt) / 2
        "Radius of the detector space"
        # Angles: uniformly spaced, n = phi_size, min = 0, max = ratio*pi
        self.angle_partition = odl.uniform_partition(0, np.pi*angle_ratio, phi_size)
        # Detector: uniformly sampled, n = 500, min = -30, max = 30
        self.detector_partition = odl.uniform_partition(-self.rho, self.rho, t_size)
    

        # Make a parallel beam geometry with flat detector
        self.geometry = odl.tomo.Parallel2dGeometry(self.angle_partition, self.detector_partition)
        self.dphi = np.mean(self.geometry.angles[1:] - self.geometry.angles[:-1])
        "Average angle step in detector"
        self.dt: float = np.mean(self.geometry.grid.meshgrid[1][0][1:] - self.geometry.grid.meshgrid[1][0][:-1])
        "Average detector step, i.e distance between adjacent detectors"

        self.ray = odl.tomo.RayTransform(self.reco_space, self.geometry)

        self.BP = BackProjection(self.ray)
    

class BasicModel(nn.Module):

    def __init__(self, geometry: Geometry, kernel: torch.Tensor = None, trainable_kernel=True, **kwargs):
        "Linear layer consisting of a 1D sinogram kernel in frequency domain"
        super(BasicModel, self).__init__(**kwargs)
        
        self.geometry = geometry
        self.BP_layer = odl_torch.OperatorModule(geometry.BP)

        if kernel == None:
            self.kernel = torch.randn((int(np.ceil(0.5 + 0.5*geometry.t_size)),)).to(DEVICE)
        else:
            assert kernel.shape == (int(np.ceil(0.5 + 0.5*geometry.t_size)),), f"wrong formatted specific kernel {kernel.shape} for geometry {geometry}"
            self.kernel = kernel.to(DEVICE)
        self.kernel.requires_grad_(trainable_kernel)
    
    def forward(self, sinos):
        filtered_sinos = torch.fft.irfft(torch.fft.rfft(sinos)*self.kernel)

        return self.BP_layer(filtered_sinos)

    def kernel_frequency_interval(self):
        "Return list with the frequencies that the model kernel values are representing"
        T_min, T_max = self.geometry.detector_partition.min_pt[0], self.geometry.detector_partition.max_pt[0]
        D = T_max - T_min
        dw = 2*np.pi / D #One or 2  2pi have to think
        return [i*dw for i in range(self.kernel.shape[0])]

    def visualize_output(self, test_sinos, test_y, loss_fn):

        ind = random.randint(0, test_sinos.shape[0]-1)
        with torch.no_grad():
            test_out = self.forward(test_sinos)  

        loss = loss_fn(test_y-test_out)
        print()
        print(f"Evaluating current kernel, validation loss: {loss.item()} using angle ratio: {self.geometry.ar}. Displayiing sample nr {ind}: ")

        sample_sino, sample_y, sample_out = test_sinos[ind].to("cpu"), test_y[ind].to("cpu"), test_out[ind].to("cpu")
        
        plt.subplot(211)
        plt.cla()
        plt.plot(self.kernel_frequency_interval(), self.kernel.detach().cpu(), label="filter in frequency domain")
        plt.legend()
        plt.subplot(223)
        plt.imshow(sample_y)
        plt.title("Real data")
        plt.subplot(224)
        plt.imshow(sample_out)
        plt.title("Filtered Backprojection")
        plt.draw()

        plt.pause(0.05)

def setup(angle_ratio = 1.0, phi_size = 100, t_size = 300, num_samples = 1000, train_ratio=0.8, pre_computed_phantoms: torch.Tensor = None):
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

    #Use stored data
    # read_data: torch.Tensor = torch.load("/data/kits_phantoms_256.pt").moveaxis(0,1)
    # read_data = torch.concat([read_data[1], read_data[0], read_data[2]])
    # read_data = read_data[:600] # -- uncomment to read this data
    read_data = torch.tensor([])

    geometry = Geometry(angle_ratio, phi_size, t_size)

    ray_layer = odl_torch.OperatorModule(geometry.ray)

    #Use previously generated phantoms to save time
    to_construct = num_samples
    if pre_computed_phantoms is None:
        pre_computed_phantoms = torch.tensor([])
    else:
        assert pre_computed_phantoms.shape[1:] == geometry.reco_space.shape
        to_construct = max(0, num_samples - pre_computed_phantoms.shape[0])
    
    #Construct new phantoms
    print("Constructing random phantoms...")
    constructed_data = np.zeros((to_construct, *geometry.reco_space.shape))
    for i in range(to_construct): #This is quite slow
        constructed_data[i] = unstructured_random_phantom(reco_space=geometry.reco_space, num_ellipses=10).asarray()

    #Combine phantoms
    permutation = list(range(pre_computed_phantoms.shape[0]))
    random.shuffle(permutation) #give this as index to tensor to randomly reshuffle order of phantoms
    full_data=torch.concat((read_data, pre_computed_phantoms[permutation], torch.from_numpy(constructed_data) )).to(DEVICE)

    print("Calculating sinograms...")
    sinos: torch.Tensor = ray_layer(full_data)

    n_training = int(num_samples*train_ratio)
    train_y, train_sinos = full_data[:n_training], sinos[:n_training]
    test_y, test_sinos = full_data[n_training:], sinos[n_training:]

    print("Constructed training dataset of shape ", train_y.shape)

    return (train_sinos, train_y, test_sinos, test_y), geometry