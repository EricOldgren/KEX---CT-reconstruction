import torch
import torch.nn as nn
import odl.contrib.torch as odl_torch

from utils.geometry import Geometry, DEVICE
from utils.fno_1d import FNO1d
from models.analyticmodels import RamLak


from math import ceil
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import  random


class CrazyKernels(nn.Module):

    reconstructionfig: Figure = None

    def __init__(self, geometry: Geometry, angle_batch_size: int) -> None:
        super().__init__()

        if angle_batch_size > 10: print("Big batch size unexpected may bahave unexpectedly")
        self.angle_batch_size = angle_batch_size
        
        self.geometry1 = geometry

        self.back1 = RamLak(geometry)

        full_phi_size  = round((geometry.phi_size * 1.0 / geometry.ar) / angle_batch_size) * angle_batch_size #maybe make cyclic in future
        self.geometry2 = Geometry(1.0, full_phi_size, geometry.t_size, reco_shape=geometry.reco_space.shape)
        modes = torch.where(self.geometry2.fourier_domain > self.geometry2.omega)[0].shape[0]
        self.ray_layer = odl_torch.OperatorModule(self.geometry2.ray)
        self.BP_layer = odl_torch.OperatorModule(self.geometry2.BP)

        self.fno = FNO1d(modes, in_channels=angle_batch_size, out_channels=angle_batch_size, dtype=torch.float).to(DEVICE)
        self.add_module("fno", self.fno)

    def forward(self, sinos: torch.Tensor):
        N, phi_size, t_size = sinos.shape

        back_bad = self.back1(sinos)
        sinos_full: torch.Tensor = self.ray_layer(back_bad)

        sinos_full = sinos_full.view(-1, self.angle_batch_size, t_size)
        sinos_full = self.fno(sinos_full)
        sinos_full = sinos_full.reshape(N, self.geometry2.phi_size, t_size)

        return self.BP_layer(sinos_full)
    
    def visualize_output(self, test_sinos, test_y, loss_fn = lambda diff : torch.mean(diff*diff), output_location = "files"):

        ind = random.randint(0, test_sinos.shape[0]-1)
        with torch.no_grad():
            test_out = self.forward(test_sinos)  
        loss = loss_fn(test_y-test_out)
        print()
        print(f"Evaluating current model state, validation loss: {loss.item()} using angle ratio: {self.geometry1.ar}. Displayiing sample nr {ind}: ")
        sample_sino, sample_y, sample_out = test_sinos[ind].to("cpu"), test_y[ind].to("cpu"), test_out[ind].to("cpu")

        if self.reconstructionfig is None:
            self.reconstructionfig, (ax_gt, ax_recon) = plt.subplots(1,2)
        else:
            ax_gt, ax_recon = self.reconstructionfig.get_axes()

        ax_gt.imshow(sample_y)
        ax_gt.set_title("Real Data")
        ax_recon.imshow(sample_out)
        ax_recon.set_title("Reconstruction")


        if output_location == "files":
            self.reconstructionfig.savefig("data/output-while-running")
            self.kernelfig.savefig("data/kernels-while-running")
            print("Updated plots saved as files")
        else:
            self.reconstructionfig.show()
            plt.show()
            self.reconstructionfig = None