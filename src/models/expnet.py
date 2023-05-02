import torch
import torch.nn as nn
import torch.nn.functional as F

import odl
import odl.contrib.torch as odl_torch

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes

import random
import os
from typing import Literal

from utils.geometry import Geometry, extend_geometry, setup, DEVICE
from models.modelbase import ModelBase
from models.fbps import FBP, GeneralizedFBP as GFBP
from models.analyticmodels import ramlak_filter, RamLak
from models.fouriernet import FNO_BP
from utils.fno_1d import FNO1d


class ExtrapolatingBP(ModelBase):

    sinofig: Figure = None

    def __init__(self, geometry: Geometry, exp_fno_layers = [30, 30], use_padding = True, fbp: ModelBase = None,  **kwargs):
        """
            Model that extrapolates sinogram and then applies an fbp operator. Extrapolaion is performed with an FNO.
            fbp defaults to an analytic RamLak operator.

            Returns fbp(Relu(FNO(X)))
        """
        super().__init__(geometry, **kwargs)
        self.use_padding = use_padding

        self.extended_geometry = extend_geometry(geometry)

        phi_size, t_size = self.geometry.phi_size, self.geometry.t_size
        ext_phi_size, _ = self.extended_geometry.phi_size, self.extended_geometry.t_size

        assert ext_phi_size > phi_size

        modes = torch.where(geometry.fourier_domain <= geometry.omega)[0].shape[0] #No padding used for fno (atm)
        self.sin2fill = FNO1d(modes, phi_size, ext_phi_size-phi_size, layer_widths=exp_fno_layers, verbose=True, dtype=torch.float).to(DEVICE)
        
        if fbp == None:       
            self.fbp = RamLak(self.extended_geometry)
        else:
            self.fbp = fbp
        self.add_module("fbp", self.fbp)

    def forward(self, X: torch.Tensor):
        fullX = self.extrapolate(X)
        return self.fbp(fullX)
    
    def extrapolate(self, X):
        N, phi_size, t_size = X.shape
        assert phi_size == self.geometry.phi_size
        filler = F.relu(self.sin2fill(X))
        assert filler.shape == (N, self.extended_geometry.phi_size-phi_size, t_size)

        return torch.concatenate([X, filler], dim=1) #full sinogram
    
    def convert(self, geometry: Geometry):
        raise NotImplementedError("this model is not convertible! (yet)")
    
    def visualize_output(self, test_sinos: torch.Tensor, test_y: torch.Tensor, full_test_sinos: torch.Tensor, loss_fn=lambda x : torch.mean(x**2), output_location: Literal["files", "show"] = "files", dirname="data", prettify_output=True):
        ind = random.randint(0, test_sinos.shape[0]-1)

        with torch.no_grad():
            exp_sinos = self.extrapolate(test_sinos)
            loss = loss_fn(full_test_sinos-exp_sinos)

        if self.sinofig is None:
            self.sinofig, (ax_gt, ax_exp) = plt.subplots(1,2)
        else:
            ax_gt, ax_exp = self.reconstructionfig.get_axes()
        print("Validation loss for sinogram extrapolation is", loss.item(), " displaying sample nr", ind)

        ax_gt.imshow(full_test_sinos[ind].cpu())
        ax_gt.set_title("Real, full sino")

        ax_exp.imshow(exp_sinos[ind].cpu())
        ax_exp.set_title("Extrapolated sino")

        self.sinofig.suptitle(f"Averagred sino MSE {loss.item()}")

        if output_location == "files":
            self.sinofig.savefig(os.path.join(dirname, "sinos-while-running"))
            self.sinofig = None
        elif output_location == "show":
            self.sinofig.show()
            self.sinofig = None
        else:
            raise ValueError(f"Invalid output location {output_location}")

        super().visualize_output(test_sinos, test_y, loss_fn, output_location, dirname, prettify_output)


if __name__ == '__main__':
    geometry = Geometry(0.5, 160, 100, reco_shape=(256, 256))

    model = ExtrapolatingBP(geometry)

    train_sinos, train_y, test_sinos, test_y = setup(geometry, num_to_generate=0, train_ratio=0.9, use_realistic=True, data_path="data/kits_phantoms_256.pt")

    model.visualize_output(test_sinos, test_y, output_location="show")

        

