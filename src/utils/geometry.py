import odl
from odl import DiscretizedSpace
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
    
    def __init__(self, angle_ratio: float, phi_size: int, t_size: int, reco_shape = (256, 256), reco_space: DiscretizedSpace = None):
        """
            Create a parallel beam geometry with corresponding forward and backward projections.
            For maximazing omega at full angle - phi_size ~ pi * t_size / 2

            parameters
                :angle_ratio - angle gemoetry is 0 to pi * angle_ratio
                :phi_size - number of angles for measurements, i.e angle resolution
                :t_size - number of lines per angle, i.e detector resolution
                :reco_shape - pixel shape of images to be reconstructed
        """
        self.ar = angle_ratio; self.phi_size = phi_size; self.t_size = t_size

        if reco_space is None:
            self.reco_space = odl.uniform_discr(min_pt=[-1.0, -1.0], max_pt=[1.0, 1.0], shape=reco_shape, dtype='float32')
        else:
            self.reco_space = reco_space
        self.rho = np.linalg.norm(self.reco_space.max_pt - self.reco_space.min_pt) / 2
        "Radius of the detector space"
        # Angles: uniformly spaced, n = phi_size, min = 0, max = angle_ratio*pi
        self.angle_partition = odl.uniform_partition(0, np.pi*angle_ratio, phi_size)
        # Detector: uniformly sampled, n = t_size, min = -rho, max = rho
        self.detector_partition = odl.uniform_partition(-self.rho, self.rho, t_size)
    
        # Make a parallel beam geometry with flat detector
        self.geometry = odl.tomo.Parallel2dGeometry(self.angle_partition, self.detector_partition)
        self.dphi = np.mean(self.geometry.angles[1:] - self.geometry.angles[:-1])
        "Average angle step in detector"
        self.dt: float = 2*self.rho / t_size
        "Average detector step, i.e distance between adjacent detectors"

        self.omega: float = np.pi * min(self.ar / (self.dphi*self.rho), 1 / self.dt) #ar added to phi term - precision is never higher than if sampling would be over full angle
        "Maximum bandwith that can be reconstructed exactly using the given partition"
        self.fourier_domain: torch.Tensor = 2*np.pi * torch.fft.rfftfreq(t_size, d=self.dt).to(DEVICE)
        "1rank tensor consisting of the frequencies where the fourier transform of functions defined on the detector partition are sampled using the discrete fourier transform"

        self.ray = odl.tomo.RayTransform(self.reco_space, self.geometry)

        self.BP = BackProjection(self.ray)
    
    def fourier_transform(self, sino):
        """
            Returns samples of the fourier transform of a function defined on the detector partition.
            Applies the torch fft on gpu and scales the result accordingly.
        """
        a = -self.rho #first sampled point in real space
        assert sino.shape[-1] == self.t_size, "Not an appropriate function"
        return self.dt*torch.exp(-1j*a*self.fourier_domain)*torch.fft.rfft(sino, axis=-1)
    
    def inverse_fourier_transform(self, sino_hat):
        "Inverse of Geometry.fourier_transform"
        a = -self.rho
        back_scaled = torch.exp(1j*a*self.fourier_domain) / self.dt * sino_hat
        return torch.fft.irfft(back_scaled, axis=-1)
    

class BasicModel(nn.Module):

    def __init__(self, geometry: Geometry, kernel: torch.Tensor = None, trainable_kernel=True, **kwargs):
        "Linear layer consisting of a 1D sinogram kernel in frequency domain"
        super(BasicModel, self).__init__(**kwargs)
        
        self.geometry = geometry
        self.BP_layer = odl_torch.OperatorModule(geometry.BP)

        if kernel == None:
            self.kernel = nn.Parameter(torch.randn(geometry.fourier_domain.shape).to(DEVICE), requires_grad=trainable_kernel)
        else:
            assert kernel.shape == geometry.fourier_domain.shape, f"wrong formatted specific kernel {kernel.shape} for geometry {geometry}"
            self.kernel = nn.Parameter(kernel.to(DEVICE), requires_grad=trainable_kernel)
    
    def forward(self, sinos):
        sino_freq = self.geometry.fourier_transform(sinos)
        filtered_sinos = self.kernel*sino_freq
        filtered_sinos = self.geometry.inverse_fourier_transform(filtered_sinos)

        return self.BP_layer(filtered_sinos)
    
    def regularisation_term(self):
        "Returns a sum which penalizies large kernel values at large frequencies, in accordance with Nattarer's sampling Theorem"
        penalty_coeffs = torch.zeros(self.geometry.fourier_domain.shape) #Create penalty coefficients -- 0 for small frequencies one above Omega
        penalty_coeffs[self.geometry.fourier_domain > self.geometry.omega] = 1.0
        
        # (mid_sec, ) = torch.where( (self.geometry.omega*0.95 < self.geometry.fourier_domain) & (self.geometry.fourier_domain <= self.geometry.omega)) # straight line joining free and panalized regions
        # penalty_coeffs[mid_sec] = torch.linspace(0, 1.0, mid_sec.shape[0])

        return torch.sum(self.kernel*self.kernel*penalty_coeffs)

    def visualize_output(self, test_sinos, test_y, loss_fn = lambda diff : torch.mean(diff*diff)):

        ind = random.randint(0, test_sinos.shape[0]-1)
        with torch.no_grad():
            test_out = self.forward(test_sinos)  

        loss = loss_fn(test_y-test_out)
        print()
        print(f"Evaluating current kernel, validation loss: {loss.item()} using angle ratio: {self.geometry.ar}. Displayiing sample nr {ind}: ")

        sample_sino, sample_y, sample_out = test_sinos[ind].to("cpu"), test_y[ind].to("cpu"), test_out[ind].to("cpu")
        
        plt.cla()
        plt.plot(self.geometry.fourier_domain.cpu(), self.kernel.detach().cpu(), label="filter in frequency domain")
        plt.legend()
        plt.figure()
        plt.subplot(121)
        plt.imshow(sample_y)
        plt.title("Real data")
        plt.subplot(122)
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
    read_data = torch.tensor([]).to(DEVICE)

    geometry = Geometry(angle_ratio, phi_size, t_size)

    ray_layer = odl_torch.OperatorModule(geometry.ray)

    #Use previously generated phantoms to save time
    to_construct = num_samples
    if pre_computed_phantoms is None:
        pre_computed_phantoms = torch.tensor([]).to(DEVICE)
    else:
        assert pre_computed_phantoms.shape[1:] == geometry.reco_space.shape
        to_construct = max(0, num_samples - pre_computed_phantoms.shape[0])
    
    #Construct new phantoms
    print("Constructing random phantoms...")
    constructed_data = np.zeros((to_construct, *geometry.reco_space.shape))
    for i in range(to_construct): #This is quite slow
        constructed_data[i] = unstructured_random_phantom(reco_space=geometry.reco_space, num_ellipses=10).asarray()
    constructed_data = torch.from_numpy(constructed_data).to(DEVICE)
    #Combine phantoms
    permutation = list(range(pre_computed_phantoms.shape[0]))
    random.shuffle(permutation) #give this as index to tensor to randomly reshuffle order of phantoms
    full_data=torch.concat((read_data, pre_computed_phantoms[permutation].to(DEVICE), constructed_data ))

    print("Calculating sinograms...")
    sinos: torch.Tensor = ray_layer(full_data)

    n_training = int(num_samples*train_ratio)
    train_y, train_sinos = full_data[:n_training], sinos[:n_training]
    test_y, test_sinos = full_data[n_training:], sinos[n_training:]

    print("Constructed training dataset of shape ", train_y.shape)

    return (train_sinos, train_y, test_sinos, test_y), geometry