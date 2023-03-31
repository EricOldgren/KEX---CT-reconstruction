import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from utils.geometry import Geometry, BasicModel as FBP, DEVICE
from typing import Literal

relu = nn.ReLU()

class FBPNet(nn.Module):

    reconstructionfig: Figure = None
    kernelfig: Figure = None

    def __init__(self, geometry: Geometry, n_fbps = 8, **kwargs):
        "2 layer network consisting of sums of FBPs"

        super(FBPNet, self).__init__(**kwargs)

        self.geometry = geometry

        self.fbps = [(FBP(geometry), nn.Parameter(torch.randn(1).to(DEVICE))) for _ in range(n_fbps)]
        self.weights = nn.Parameter(torch.randn(n_fbps).to(DEVICE))
        self.bout = nn.Parameter(torch.randn(1).to(DEVICE))

        for i, (fbp, b) in enumerate(self.fbps):
            self.add_module(f"fbp{i}", fbp)
            self.register_parameter(f"b{i}", b)


    def forward(self, x):
        constructs = torch.stack([relu(fbp(x) + b) for fbp, b in self.fbps])    #memory problem

        out = torch.sum(constructs*self.weights[:, None, None, None], axis=0)

        return relu(out + self.bout)

    def regularization_term(self):
        "Returns a sum which penalizies large kernel values at large frequencies, in accordance with Nattarer's sampling Theorem"
        
        return sum(self.weights[i]*self.weights[i]*self.fbps[i][0].regularisation_term() for i in range(len(self.fbps)) ) / len(self.fbps)

    def convert(self, geometry: Geometry):
        m2 = FBPNet(geometry, n_fbps=len(self.fbps))
        m2.load_state_dict(self.state_dict()) #loads weights, biases and kernels -- however kernel loading may be incompatible
        for i in range(len(self.fbps)):
            m2.fbps[i] = (self.fbps[i][0].convert(geometry), self.fbps[i][1]) #this will raise error if incompatible
            
        return m2
        
        

    def visualize_output(self, test_sinos, test_y, loss_fn = lambda diff : torch.mean(diff*diff), output_location = "files"):

        ind = random.randint(0, test_sinos.shape[0]-1)
        with torch.no_grad():
            test_out = self.forward(test_sinos)  #memory problem
        loss = loss_fn(test_y-test_out)
        print()
        print(f"Evaluating current model state, validation loss: {loss.item()} using angle ratio: {self.geometry.ar}. Displayiing sample nr {ind}: ")
        sample_sino, sample_y, sample_out = test_sinos[ind].to("cpu"), test_y[ind].to("cpu"), test_out[ind].to("cpu")

        if self.reconstructionfig is None:
            self.reconstructionfig, (ax_gt, ax_recon) = plt.subplots(1,2)
        else:
            ax_gt, ax_recon = self.reconstructionfig.get_axes()

        ax_gt.imshow(sample_y)
        ax_gt.set_title("Real Data")
        ax_recon.imshow(sample_out)
        ax_recon.set_title("Reconstruction")

        self.draw_kernels()

        if output_location == "files":
            self.reconstructionfig.savefig("output-while-running")
            self.kernelfig.savefig("kernels-while-running")
            print("Updated plots saved as files")
        else:
            plt.draw()
            self.reconstructionfig.show()
            self.kernelfig.show()
            plt.show()
            self.reconstructionfig = None
            self.kernelfig = None

        
    
    def draw_kernels(self):
        
        if self.kernelfig is None:
            self.kernelfig, ax = plt.subplots(1,1)
        else:
            ax, = self.kernelfig.get_axes()
        ax.cla()
        for i, (fbp, b) in enumerate(self.fbps):
            ax.plot(fbp.geometry.fourier_domain.cpu(), fbp.kernel.detach().cpu(), label=f"filter {i}")
        m, M = ax.get_ylim(); horizontal = np.linspace(m, M, 30)
        ax.plot([self.geometry.omega]*horizontal.shape[0], horizontal, dashes=[2,2], c='#000', label="omega")

        ax.legend(loc="lower left")

