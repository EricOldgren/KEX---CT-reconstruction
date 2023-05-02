import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from typing import Literal, Mapping, Any, List
from utils.geometry import Geometry, DEVICE
import odl.contrib.torch as odl_torch
import numpy as np
import os

class ModelBase(nn.Module):

    reconstructionfig: Figure = None
    kernelfig: Figure = None
    plotkernels = False

    use_padding = False

    def __init__(self, geometry: Geometry, **kwargs):
        super().__init__(**kwargs)
        self.geometry = geometry
        self.BP_layer = odl_torch.OperatorModule(geometry.BP)

    def kernels(self)->'list[torch.Tensor]':
        """
            Return list of kernels used by the model.
        """
        raise NotImplementedError(f"{type(self)} has not implemented a kernels method but is using it. The modelbase function should be overwritten.")

    def regularization_term(self):
        raise Warning(f"{self.__class__} has no regularization implemented but the function is called. This should be overwritten.")
        return 0.0

    def state_dict(self, **kwargs):
        sd = super().state_dict(**kwargs)
        for name, subm in self.named_modules():
            if subm is not self and isinstance(subm, ModelBase):
                sub_sd = subm.state_dict()
                for k, v in sub_sd.items(): sd[f"{name}.{k}"] = v
                
        sd["ar"] = self.geometry.ar; sd["phi_size"] = self.geometry.phi_size; sd["t_size"] = self.geometry.t_size
        return sd

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        if strict: assert self.state_dict().keys() == state_dict.keys()
        ar, phi_size, t_size = state_dict['ar'], state_dict['phi_size'], state_dict['t_size']
        super().load_state_dict(state_dict, strict=False) #loads weights, biases and kernels -- not geometry
        geometry = Geometry(ar, phi_size, t_size)
        self.geometry = geometry
        self.BP_layer = odl_torch.OperatorModule(geometry.BP)
    
    @classmethod
    def model_from_state_dict(clc, state_dict, use_padding=True):
        ar, phi_size, t_size = state_dict['ar'], state_dict['phi_size'], state_dict['t_size']
        g = Geometry(ar, phi_size, t_size)
        m = clc(g, use_padding=use_padding)
        m.load_state_dict(state_dict)
        return m

    def convert(self, geometry: Geometry):
        "Return a new model with the same kernels that reconstructs sinograms from the given geometry. This converts all submodels of type ModelBase."
        assert (geometry.fourier_domain == self.geometry.fourier_domain).all(), f"Converting requires geometries to have the same fourier domain. That is guaranteed if they have the same t_size and rho."
        sd = self.state_dict()
        sd["ar"] = geometry.ar; sd["phi_size"] = geometry.phi_size; sd["t_size"] = geometry.t_size
        for name, submodule in self.named_modules():
            if not submodule is self and isinstance(submodule, ModelBase):
                sd[f"{name}.ar"] = geometry.ar;  sd[f"{name}.phi_size"] = geometry.phi_size; sd[f"{name}.t_size"] = geometry.t_size
        m2 = self.__class__.model_from_state_dict(sd)

        return m2

    def visualize_output(self, test_sinos: torch.Tensor, test_y: torch.Tensor, loss_fn = lambda diff : torch.mean(diff*diff), output_location: Literal["files", "show"] = "files", dirname = "data", prettify_output = True):
        """
            Evaluates loss with respect to input and displays a random sample.

            parameters
                - test_sinos: (batch_size x phi_size x t_size) batch of sinograms
                - test_y: (batch_size x reco_shape) batch of corresponding image data
                - loss_fn: loss function to use to evaluate loss
                - output_location: whether to show plots or store to files
                - dirname: only used when output_location = files, path to folder where plots are stored - names are output-while-running and kernels-while-running
                - prettify_output (bool): if True pixel values larger than the maximum of the ground truth are truncated to maintain the same color scaling in the gt and recon images
        """
        ind = random.randint(0, test_sinos.shape[0]-1)
        with torch.no_grad():
            test_out = self.forward(test_sinos)  
        loss = loss_fn(test_y-test_out)
        print()
        print(f"Evaluating current model state, validation loss: {loss.item()}. Displayiing sample nr {ind}: ")
        sample_sino, sample_y, sample_out = test_sinos[ind].to("cpu"), test_y[ind].to("cpu"), test_out[ind].to("cpu")
        if prettify_output:
            sample_out = torch.minimum(sample_out, torch.ones(sample_out.shape, device="cpu")*torch.max(sample_y))

        if self.reconstructionfig is None:
            self.reconstructionfig, (ax_gt, ax_recon) = plt.subplots(1,2)
        else:
            ax_gt, ax_recon = self.reconstructionfig.get_axes()

        self.reconstructionfig.suptitle(f"Output. Averaged MSE {loss.item()}")
        ax_gt.imshow(sample_y)
        ax_gt.set_title("Real Data")
        ax_recon.imshow(sample_out)
        ax_recon.set_title("Reconstruction")
        
        if self.plotkernels: self.draw_kernels()

        if output_location == "files":
            self.reconstructionfig.savefig(os.path.join(dirname, "output-while-running"))
            if self.plotkernels: self.kernelfig.savefig(f"{os.path.join(dirname, 'kernels-while-running')}")
            self.reconstructionfig = None
            self.kernelfig = None
            print("Updated plots saved as files")
        elif output_location == "show":
            self.reconstructionfig.show()
            if self.plotkernels: self.kernelfig.show()
            plt.show()
            self.reconstructionfig = None
            self.kernelfig = None
        else:
            raise ValueError(f"Invalid output location {output_location}")
        
    def draw_kernels(self):
        "Plot kernels of model"
        if not self.plotkernels:
            return
        kernels = self.kernels()
        is_complex = kernels[0].dtype.is_complex
        if self.kernelfig is None:
            if is_complex: self.kernelfig, axs = plt.subplots(2,1)
            else: self.kernelfig, axs = plt.subplots(1,1)
        else:
            axs, = self.kernelfig.get_axes()
        if not isinstance(axs, np.ndarray): axs = np.array([axs]) #axs should be list of axes, plt.subplots returns only an axes object if called with 1 x 1 layout.
        self.kernelfig.suptitle("Kernels")
        omgs = self.geometry.fourier_domain if not self.use_padding else self.geometry.fourier_domain_padded

        for ax in axs:
            ax: Axes
            ax.cla()
            for i, kernel in enumerate(kernels):
                if ax is axs[0]:
                    kernel = kernel.real
                    ax.set_title("Real Part")
                else:
                    kernel = (-1j*kernel).real
                    ax.set_title("Imaginary Part")
                ax.plot(omgs.cpu(), kernel.detach().cpu(), label=f"filter {i}")
            m, M = ax.get_ylim(); horizontal = np.linspace(m, M, 30)
            ax.plot([self.geometry.omega]*horizontal.shape[0], horizontal, dashes=[2,2], c='#000', label="omega")

        ax.legend(loc="lower left")
        

class ChainedModels(ModelBase):

    def __init__(self, *models: ModelBase, **kwargs):
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

    def state_dict(self, **kwargs):
        return super(ModelBase, self).state_dict(**kwargs)
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        super(ModelBase, self).load_state_dict(state_dict, strict)
        self.rays = []
        for i, model in enumerate(self.models):
            if i < len(self.models)-1:
                self.rays.append(odl_torch.OperatorModule(model.geometry.ray))
        self.rays.append(None)
    
    def convert(self, geometry: Geometry):
        "Converts model chain to reconstruct from another geometry. This will only convert the first model in the chain."
        models = [self.models[0].convert(geometry)] + [m.convert(m.geometry) for m in self.models[1:]]
        return ChainedModels(models)
    