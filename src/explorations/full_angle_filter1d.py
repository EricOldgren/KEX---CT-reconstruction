import odl
import odl.contrib.torch as odl_torch
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Dataset
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import win32api

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
reco_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20], shape=[128, 128], dtype='float32')
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
    filtered_back_projected = BP(filtered_sino)
    plt.imshow(filtered_back_projected)
    plt.title("Filtered Backprojection")
    plt.draw()

def dirac_approx(l, dx=1):
    rh = torch.tensor([1/(i*dx) for i in range(1, l+1)])
    lh = torch.flip(rh, (0, ))
    return torch.concat([lh, torch.tensor([1]), rh])


if __name__ == '__main__':

    # full_data: torch.Tensor = torch.load("kits_phantoms_256.pt").moveaxis(0,1)
    # full_data = torch.concat([full_data[1], full_data[0], full_data[2]])
    # train_y = full_data[:600]
    # test_sample = full_data[601]

    full_data = []
    for _ in range(600):
        # full_data.append(odl.phantom.transmission.forbild(reco_space, resolution=True).asarray())
        full_data.append(odl.phantom.transmission.shepp_logan(reco_space, True).asarray())
    full_data = torch.from_numpy(np.array(full_data))
    train_y = full_data[:200]
    test_sample = full_data[40]
    print("Test ranfe: ", torch.min(test_sample), torch.max(test_sample))

    print("Training data shape ", train_y.shape)

    print("Calculating sinograms...")
    train_sinos = ray_layer(train_y)
    phi_size, t_size = train_sinos.shape[1:]

    #Simple CNN
    kernel = torch.randn((1, 1, 1, 11), dtype=torch.float32)*0.1 #1d kernel dimensions represent : 1outchannel x 1inchannel x height=1 x full_width
    dirac_res = 4
    # kernel[0, 0, 0, int(0.5*11)-dirac_res:int(0.5*11)+dirac_res+1] = dirac_approx(dirac_res, dx=1.0)
    kernel[0, 0, 0, int(11*0.5)] = 0.5
    CONVOLVE = lambda sino_batch, kernel : nn.functional.conv2d(sino_batch[:, None], kernel, padding="same")[:, 0]

    kernel.requires_grad_(True)
    optimizer = torch.optim.SGD([kernel], 0.0003)
    loss_fn = nn.L1Loss()

    N_epochs = 10
    dataloader = DataLoader(list(zip(train_sinos, train_y)), batch_size=30, shuffle=True)

    def sigint_handler(signal_code, hook=None):
        if signal_code == 0: #Ctr+C
            print("Saving Kernel and sample projection...")
            torch.save(kernel, "latest_kernel.pt")
            plt.savefig("Latest_Reconstruction_Plots")
    win32api.SetConsoleCtrlHandler(sigint_handler, 1)

    for epoch in range(N_epochs):
        # pbar = tqdm(dataloader)
        for data_batch in dataloader:
            update_display(test_sample, kernel.detach().numpy().copy(), CONVOLVE)
            plt.pause(0.05) #pyplot needs time to update GUI

            sino_batch, y_batch = data_batch
            filtered = CONVOLVE(sino_batch, kernel)
            out = BP_layer(filtered)

            loss = loss_fn(out, y_batch)
            # pbar.set_description(f"Epoch {epoch}, loss={loss.item()}")
            print(f"Epoch {epoch}, loss={loss.item()}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print()