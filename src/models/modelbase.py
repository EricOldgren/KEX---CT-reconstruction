import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Literal, Mapping, Any, List
from utils.geometry import Geometry
import odl.contrib.torch as odl_torch

class ModelBase(nn.Module):

    reconstructionfig: Figure = None

    def __init__(self, geometry: Geometry, **kwargs):
        super().__init__(**kwargs)
        self.geometry = geometry
        self.BP_layer = odl_torch.OperatorModule(geometry.BP)

    def state_dict(self):
        sd = super().state_dict()
        sd["ar"] = self.geometry.ar; sd["phi_size"] = self.geometry.phi_size; sd["t_size"] = self.geometry.t_size
        return sd

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        ar, phi_size, t_size = state_dict['ar'], state_dict['phi_size'], state_dict['t_size']
        super_states = {k: v for k, v in state_dict.items() if k not in ("ar", "phi_size", "t_size")}
        super().load_state_dict(super_states, strict) #loads weights, biases and kernels -- however kernel loading may be incompatible
        geometry = Geometry(ar, phi_size, t_size)
        self.geometry = geometry
        self.BP_layer = odl_torch.OperatorModule(geometry.BP)

    def visualize_output(self, test_sinos: torch.Tensor, test_y: torch.Tensor, loss_fn = lambda diff : torch.mean(diff*diff), output_location: Literal["files", "show"] = "files", file_location = "data/output-while-running"):
        """
            Evaluates loss with respect to input and displays a random sample.

            parameters
                - test_sinos: (batch_size x phi_size x t_size) batch of sinograms
                - test_y: (batch_size x reco_shape) batch of corresponding image data
                - loss_fn: loss function to use to evaluate loss
                - output_location: whether to show plots or store to files
                - file_location: only used when output_location = files, path to file where output is stored
        """
        ind = random.randint(0, test_sinos.shape[0]-1)
        with torch.no_grad():
            test_out = self.forward(test_sinos)  
        loss = loss_fn(test_y-test_out)
        print()
        print(f"Evaluating current model state, validation loss: {loss.item()}. Displayiing sample nr {ind}: ")
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
            self.reconstructionfig.savefig(file_location)
            self.reconstructionfig = None
            print("Updated plots saved as files")
        elif output_location == "show":
            self.reconstructionfig.show()
            plt.show()
            self.reconstructionfig = None
        else:
            raise ValueError(f"Invalid output location {output_location}")

class ChainedModels(ModelBase):

    def __init__(self, models: List[ModelBase], **kwargs):
        "Chains a list of back projection models into a multilayer network"
        super(ModelBase, self).__init__(**kwargs)

        self.models = models
        self.rays = []
        for i, model in enumerate(models):
            self.add_module(f"model {i}", model)
            if i < len(models)-1:
                self.rays.append(odl_torch.OperatorModule(models[i+1].geometry.ray))
        self.rays.append(None) #ends in image domain
    
    def forward(self, X):
        out = X
        for model, ray in zip(self.models, self.rays):
            out = F.relu(model(out))
            if ray is not None:
                out = ray(out)
        
        return out

    def state_dict(self):
        return super(ModelBase, self).state_dict()
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        super(ModelBase, self).load_state_dict(state_dict, strict)
        self.rays = []
        for i, model in enumerate(self.models):
            if i < len(self.models)-1:
                self.rays.append(odl_torch.OperatorModule(model.geometry.ray))
        self.rays.append(None)
    