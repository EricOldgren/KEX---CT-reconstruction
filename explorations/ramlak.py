import odl
import odl.contrib.torch as odl_torch
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Dataset
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

class BackProjection(odl.Operator):

    def __init__(self, reco_space, geometry):

        self.ray = odl.tomo.RayTransform(reco_space, geometry)
        super().__init__(self.ray.range, self.ray.domain, linear=True)
    
    def _call(self, x):
        return self.ray.adjoint(x)

    @property
    def adjoint(self):
        return self.ray

# Reconstruction space: discretized functions on the rectangle [-20, 20]^2 with 300 samples per dimension.
reco_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20], shape=[256, 256], dtype='float32')

#sinogram resolution
phi_size, t_size = 100, 300
# Angles: uniformly spaced, n = 1000, min = 0, max = pi
angle_partition = odl.uniform_partition(0, np.pi, phi_size)
# Detector: uniformly sampled, n = 500, min = -30, max = 30
detector_partition = odl.uniform_partition(-30, 30, t_size)

# Make a parallel beam geometry with flat detector
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
ray = odl.tomo.RayTransform(reco_space, geometry)
ray_layer = odl_torch.OperatorModule(ray)

BP = BackProjection(reco_space, geometry)
BP_layer = odl_torch.OperatorModule(BP)

def update_display(test_sample, kernel, CONVOLVE):
    kernel = torch.tensor(kernel)
    plt.subplot(221)
    plt.imshow(test_sample)
    plt.title("Real data")
    plt.subplot(422)
    sample_sino = ray_layer(test_sample[None])[0]
    plt.imshow(sample_sino)
    plt.title("Sinograms")
    plt.subplot(424)
    filtered_sino = CONVOLVE(sample_sino[None], kernel)[0]
    plt.imshow(filtered_sino)
    plt.subplot(223)
    f = kernel[0,0,0]
    k = torch.fft.fft(f)
    plt.cla()
    plt.plot(list(range(f.shape[0])), k, label="fourier transformed filter")
    plt.plot(list(range(f.shape[0])), f, label="filter")
    plt.legend()
    plt.subplot(224)
    filtered_back_projected = BP(sample_sino)
    plt.imshow(filtered_back_projected)
    plt.title("Filtered Backprojection")
    plt.draw()


def fbp(sino, freq_ker):
    f_sino = torch.fft.rfft(sino, dim=-1)
    filtered_sino = torch.fft.irfft(f_sino*freq_ker, dim=-1)
    return BP_layer(filtered_sino[None])[0]


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    full_data: torch.Tensor = torch.load(os.path.join(os.path.dirname(__file__), "kits_phantoms_256.pt")).moveaxis(0,1)
    full_data = torch.concat([full_data[1], full_data[0], full_data[2]]).to(device)

    # full_data = []
    # for _ in range(100):
    #     full_data.append(odl.phantom.transmission.forbild(reco_space, resolution=True).asarray())
    #     # full_data.append(odl.phantom.transmission.shepp_logan(reco_space, True).asarray())
    # full_data = torch.from_numpy(np.array(full_data)).to(device)
    train_y = full_data[:20]
    test_sample = full_data[21]
    print("Test range: ", torch.min(test_sample), torch.max(test_sample))

    print("Training data shape ", train_y.shape)

    print("Calculating sinograms...")
    train_sinos = ray_layer(train_y)

    kernel_r_freq = torch.arange(np.ceil(0.5 + 0.5*t_size))
    # kernel_r_freq[int(0.25*t_size):] = 0
    
    loss_fn = lambda diff : torch.mean(torch.abs(diff))
    dataloader = DataLoader(list(zip(train_sinos, train_y)), batch_size=100, shuffle=True)
    

    for sino_batch, real_deta_batch in dataloader:
        f_sino = torch.fft.rfft(sino_batch, dim=-1)
        filtered_sino_batch = torch.fft.irfft(f_sino*kernel_r_freq, dim=-1)
        out_batch = BP_layer(filtered_sino_batch)
        out_batch /= torch.max(out_batch.view(out_batch.shape[0], -1), dim=-1, keepdim=True).values[:, :, None] #Normalize
        loss = loss_fn(out_batch-real_deta_batch)

        print(torch.max(real_deta_batch))
        print(torch.max(out_batch))
        print(torch.max(out_batch-real_deta_batch))
        print(f"loss={loss}")
        print()

    test_sino = ray_layer(test_sample[None])[0]
    outp = fbp(test_sino, kernel_r_freq).detach().to("cpu")
    outp /= torch.max(outp)
    print("Test stats")
    print(f"loss={loss_fn(outp-test_sample)}")
    print(torch.max(outp), torch.max(test_sample))
    plt.subplot(221)
    plt.imshow(test_sample)
    plt.title("Real data")
    plt.subplot(422)
    sample_sino = ray_layer(test_sample[None])[0]
    plt.imshow(sample_sino)
    plt.title("Sinograms")
    plt.subplot(424)
    filtered_sino = torch.fft.irfft(torch.fft.rfft(sample_sino)*kernel_r_freq)
    plt.imshow(filtered_sino)
    plt.subplot(223)
    plt.plot([i for i in range(len(kernel_r_freq))], kernel_r_freq)
    plt.subplot(224)
    outp = BP_layer(filtered_sino[None])[0]
    plt.imshow(outp)
    plt.title("Filtered backprojection with ram lak filter")
    plt.show()


    plt.subplot(223)
    plt.plot([i for i in range(len(kernel_r_freq))], kernel_r_freq)
    plt.subplot(224)
    sino_kernel = torch.fft.irfft(kernel_r_freq)
    sino_half = sino_kernel[:int(0.5*t_size)]
    sino_kernel = torch.concat([torch.flip(sino_half, dims=(0,)), sino_half, torch.zeros(t_size - 2*int(t_size*0.5))])
    plt.plot([i for i in range(len(sino_kernel))], sino_kernel)
    plt.show()




    sino_kernel = sino_kernel.repeat((phi_size, 1))
    print(sino_kernel.shape)
    ram_filter = kernel_r_freq

    space_filter = BP_layer(sino_kernel[None])[0]
    freq_filter_r = torch.fft.fft2(space_filter[128:,128:]).real
    freq_filter_i = torch.fft.fft2(space_filter[128:, 128:]).imag

    print()
    print("Space filter stats")
    print("in image domain")
    print("shape ", space_filter.shape, " mean: ", torch.mean(space_filter), " std: ", torch.std(space_filter), " max: ", torch.max(space_filter))
    print("in frequency domain: ")
    print("shape: ", freq_filter_r.shape, " real max: ", torch.max(freq_filter_r), " imaginary max: ", torch.max(freq_filter_i), " real std: ", torch.std(freq_filter_r))
    plt.subplot(121)
    plt.title("Image filter, pixel domain")
    plt.imshow(space_filter)
    plt.subplot(122)
    plt.imshow(freq_filter_r)
    plt.title("Image filter in frequency domain")
    plt.show()

    
