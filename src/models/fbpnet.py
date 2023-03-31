import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from utils.geometry import Geometry, BasicModel, DEVICE
from utils.smoothing import SmoothedModel
from typing import Literal
from typing import Mapping, Any


class FBPNet(nn.Module):

    reconstructionfig: Figure = None
    kernelfig: Figure = None

    def __init__(self, geometry: Geometry, n_fbps = 8, use_smooth_filters = False, **kwargs):
        "2 layer network consisting of sums of FBPs"

        super(FBPNet, self).__init__(**kwargs)

        self.geometry = geometry
        if use_smooth_filters:
            self.fbps = [(SmoothedModel(geometry), nn.Parameter(torch.randn(1).to(DEVICE))) for _ in range(n_fbps)]
        else:
            self.fbps = [(BasicModel(geometry), nn.Parameter(torch.randn(1).to(DEVICE))) for _ in range(n_fbps)]

        self.weights = nn.Parameter(torch.randn(n_fbps).to(DEVICE))
        self.bout = nn.Parameter(torch.randn(1).to(DEVICE))
        self.relu = nn.ReLU()

        for i, (fbp, b) in enumerate(self.fbps):
            self.add_module(f"fbp{i}", fbp)
            self.register_parameter(f"b{i}", b)


    def forward(self, x):
        constructs = torch.stack([self.relu(fbp(x) + b) for fbp, b in self.fbps])

        out = torch.sum(constructs*self.weights[:, None, None, None], axis=0)

        return self.relu(out)

    def regularization_term(self):
        "Returns a sum which penalizies large kernel values at large frequencies, in accordance with Nattarer's sampling Theorem"
        
        return sum(self.weights[i]*self.weights[i]*self.fbps[i][0].regularisation_term() for i in range(len(self.fbps)) ) / len(self.fbps)

    def convert(self, geometry: Geometry):
        m2 = FBPNet(geometry, n_fbps=len(self.fbps))
        sd = self.state_dict()
        sd["ar"] = geometry.ar; sd["phi_size"] = geometry.phi_size; sd["t_size"] = geometry.t_size
        m2.load_state_dict(sd) 
        for i in range(len(self.fbps)):
            m2.fbps[i] = (self.fbps[i][0].convert(geometry), self.fbps[i][1]) #this will raise error if incompatible
            
        return m2
        
    def state_dict(self):
        sd = super().state_dict()
        sd["ar"] = self.geometry.ar; sd["phi_size"] = self.geometry.phi_size; sd["t_size"] = self.geometry.t_size
        sd["n_fbps"] = len(self.fbps)
        return sd

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        ar, phi_size, t_size = state_dict['ar'], state_dict['phi_size'], state_dict['t_size']
        assert state_dict["n_fbps"] == len(self.fbps), f"Incompatible state dict"
        super_states = {k: v for k, v in state_dict.items() if k not in ("ar", "phi_size", "t_size", "n_fbps")}
        super().load_state_dict(super_states, strict) #loads weights, biases and kernels -- however kernel loading may be incompatible
        geometry = Geometry(ar, phi_size, t_size)
        self.geometry = geometry

        for i in range(len(self.fbps)):
            self.fbps[i] = (self.fbps[i][0].convert(geometry), self.fbps[i][1]) #this will raise error if incompatible

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
            self.reconstructionfig.savefig("data/output-while-running")
            self.kernelfig.savefig("data/kernels-while-running")
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

def load_fbpnet_from_dict(path, smooth_filters: bool = False):
    sd = torch.load(path)
    ar, phi_size, t_size = sd["ar"], sd["phi_size"], sd["t_size"]
    geometry = Geometry(ar, phi_size, t_size)
    ret = FBPNet(geometry, n_fbps=sd["n_fbps"], use_smooth_filters=smooth_filters)
    ret.load_state_dict(sd)
    return ret

