import odl
import odl.contrib.torch as odl_torch
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Dataset
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import signal

from data_generator import random_ellipsoid,random_phantom

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
# Angles: uniformly spaced, n = 1000, min = 0, max = pi
angle_partition = odl.uniform_partition(0, np.pi, 100)
# Detector: uniformly sampled, n = 500, min = -30, max = 30
detector_partition = odl.uniform_partition(-30, 30, 300)

# Make a parallel beam geometry with flat detector
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
ray = odl.tomo.RayTransform(reco_space, geometry)
ray_layer = odl_torch.OperatorModule(ray)

BP = BackProjection(reco_space, geometry)
BP_layer = odl_torch.OperatorModule(BP)

def update_display(test_sample, kernel_freq):
    plt.subplot(221)
    plt.imshow(test_sample)
    plt.title("Real data")
    plt.subplot(422)
    sample_sino = ray_layer(test_sample[None])[0]
    plt.imshow(sample_sino)
    plt.title("Sinograms")
    plt.subplot(424)
    filtered_sino = torch.fft.irfft(torch.fft.rfft(sample_sino, dim=-1)*kernel_freq)
    plt.imshow(filtered_sino)
    plt.subplot(223)
    plt.cla()
    plt.plot(list(range(kernel_freq.shape[0])), kernel_freq, label="filter in frequency domain")
    plt.legend()
    plt.subplot(224)
    filtered_back_projected = BP_layer(filtered_sino[None])[0]
    plt.imshow(filtered_back_projected)
    plt.title("Filtered Backprojection")
    plt.draw()


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    full_data: torch.Tensor = torch.load("Phantom_data\kits_phantoms_256.pt").moveaxis(0,1)
    full_data = torch.concat([full_data[1], full_data[0], full_data[2]])

    additional_data=[]
    for i in range(1):
        additional_data.append(random_phantom(reco_space=reco_space,num_ellipses=7))
    full_data=torch.concat((full_data,torch.tensor(np.array(additional_data)))).to(device)
    train_y = full_data[:500]

    #full_data = []
    #for _ in range(600):
        #full_data.append(odl.phantom.transmission.forbild(reco_space, True).asarray())
    #full_data = torch.from_numpy(np.array(full_data)).to(device)

    test_sample = full_data[301]
    print("Test ranfe: ", torch.min(test_sample), torch.max(test_sample))

    print("Training data shape ", train_y.shape)

    print("Calculating sinograms...")
    train_sinos = ray_layer(train_y)
    phi_size, t_size = train_sinos.shape[1:]
    
    kernel_freq = torch.arange(np.ceil(0.5 + 0.5*t_size)).to(device)
    kernel_freq = torch.randn(kernel_freq.shape).to(device)

    kernel_freq.requires_grad_(True)
    optimizer = torch.optim.Adam([kernel_freq], lr=0.003)
    loss_fn = lambda diff : torch.mean(torch.abs(diff))

    N_epochs = 100
    dataloader = DataLoader(list(zip(train_sinos, train_y)), batch_size=30, shuffle=True)

    for epoch in range(N_epochs):
        # pbar = tqdm(dataloader)
        if epoch%5==0:
            update_display(test_sample.to("cpu"), kernel_freq.detach().to("cpu"))
            plt.pause(0.05) #pyplot needs time to update GUI
        for data_batch in dataloader:

            sino_batch, y_batch = data_batch
            sino_freq = torch.fft.rfft(sino_batch, dim=-1)

            filtered_sinos = torch.fft.irfft(sino_freq*kernel_freq, dim=-1)
            out = BP_layer(filtered_sinos)
            # out /= torch.max(out.view(out.shape[0], -1), dim=-1, keepdim=True).values[:, :, None] #Normalize

            loss = loss_fn(out-y_batch)
            # pbar.set_description(f"Epoch {epoch}, loss={loss.item()}")
            print(f"Epoch {epoch}, loss={loss.item()}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print()
    
    print("Saving Kernel and sample projection...")
    torch.save(kernel_freq, "latest_kernel_frequency_domain.pt")
    plt.savefig("Latest_Reconstruction_Plots_frequencyfilter")
    
