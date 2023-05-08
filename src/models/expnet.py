import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize

from math import ceil
import numpy as np

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
from models.fouriernet import FNO_BP, GeneralizedFNO_BP as GFNO_BP, fno_from_sd
from utils.fno_1d import FNO1d
from utils.moments import SinoMoments

class ExtrapolatingBP(ModelBase):
    """
            Model that extrapolates sinogram and then applies an fbp operator.
            fbp defaults to an analytic RamLak operator.

            Returns fbp(Relu(sin2fill(X)))
    """

    sinofig: Figure = None

    def __init__(self, geometry: Geometry, sin2filler: nn.Module = None, sin2full: nn.Module = None, extended_geometry: Geometry = None, fbp: 'ModelBase|str' = None, use_padding = True, **kwargs):
        "Specify sin2fill layer"
       
        super().__init__(geometry, **kwargs)
        if isinstance(sin2filler, nn.Module):
            self.sin2filler = sin2filler
            self.sin2full = None
        else:
            assert isinstance(sin2full, nn.Module)
            self.sin2full = sin2full
            self.sin2filler = None
        if extended_geometry is None: extended_geometry = extend_geometry(geometry)
        self.extended_geometry = extended_geometry

        if fbp is None or (isinstance(fbp, str) and fbp.lower() in ("ramlak", "ram-lak", "analytic")):
            self.fbp = RamLak(self.extended_geometry, use_padding=use_padding)
        elif isinstance(fbp, str)  and fbp.lower() == "fno":
            ext_modes = torch.where(self.extended_geometry.fourier_domain <= geometry.omega)[0].shape[0]
            ext_fno = FNO1d(ext_modes, self.extended_geometry.phi_size, self.extended_geometry.phi_size, layer_widths=[30, 30], verbose=True, dtype=torch.float)
            self.fbp = GFNO_BP(self.extended_geometry, ext_fno, use_padding=use_padding)
        else:
            self.fbp = fbp
    
    def extrapolate(self, X):
        N, phi_size, t_size = X.shape
        if self.sin2full == None:
            assert phi_size == self.geometry.phi_size
            filler = F.relu(self.sin2filler(X))
            assert filler.shape == (N, self.extended_geometry.phi_size-phi_size, t_size)

            return torch.concatenate([X, filler], dim=1) #full sinogram
        
        return F.relu(self.sin2full(X))

    def forward(self, X: torch.Tensor):
        fullX = self.extrapolate(X)
        return self.fbp(fullX)

    def convert(self, geometry: Geometry):
        raise NotImplementedError("this model is not convertible! (yet)")
    
    def visualize_output(self, test_sinos: torch.Tensor, test_y: torch.Tensor, full_test_sinos: torch.Tensor, output_location: Literal["files", "show"] = "files", dirname="data", prettify_output=True, ind = None):
        if ind is None: ind = random.randint(0, test_sinos.shape[0]-1)
        mse_fn = lambda diff: torch.mean(diff**2)


        exp_sinos = self.extrapolate(test_sinos).detach()
        recons = self.fbp(exp_sinos)
        mse_sinos = mse_fn(full_test_sinos-exp_sinos)
        mse_recons = mse_fn(test_y-recons)

        print("Validation loss for sinogram extrapolation is", mse_sinos.item(), "displaying sample nr", ind)
        print("Validation loss for reconstruction is", mse_recons.item(), "displaying sample nr", ind)

        if self.sinofig is None:
            self.sinofig, (sino_ax_gt, sino_ax_exp) = plt.subplots(1,2)
        else:
            sino_ax_gt, sino_ax_exp = self.reconstructionfig.get_axes()
        if self.reconstructionfig is None:
            self.reconstructionfig, (img_ax_gt, img_ax_recon) = plt.subplots(1,2)
        else:
            img_ax_gt, img_ax_recon = plt.subplots(1,2)

        sino_ax_gt.imshow(full_test_sinos[ind].cpu())
        sino_ax_gt.set_title("full sino GT")
        sino_ax_exp.imshow(exp_sinos[ind].detach().cpu())
        sino_ax_exp.set_title("Extrapolated sino")
        self.sinofig.suptitle(f"Averagred sino MSE {mse_sinos.item()}")

        img_ax_gt.imshow(test_y[ind].cpu())
        img_ax_gt.set_title("GT")
        img_ax_recon.imshow(recons[ind].detach().cpu())
        img_ax_recon.set_title("Reconstruction")
        self.reconstructionfig.suptitle(f"Averaged recon MSE {mse_recons.item()}")

        if self.plotkernels: self.draw_kernels()

        if output_location == "files":
            self.sinofig.savefig(os.path.join(dirname, "sinos-while-running"))
            self.reconstructionfig.savefig(os.path.join(dirname, "output-while-running"))
            if self.plotkernels: self.kernelfig.savefig(f"{os.path.join(dirname, 'kernels-while-running')}")
            print("Updated plots saved as files")
        elif output_location == "show":
            self.sinofig.show()
            self.reconstructionfig.show()
            if self.plotkernels: self.kernelfig.show()
            plt.show()
        else:
            raise ValueError(f"Invalid output location {output_location}")
        self.sinofig = None
        self.reconstructionfig = None
        self.kernelfig = None

class FNOExtrapolatingBP(ExtrapolatingBP):
    """
        Model that extrapolates sinograms using an FNO.
    """

    def __init__(self, geometry: Geometry, exp_fno_layers = [30,30], fbp: ModelBase | str = None, **kwargs):

        extended_geometry = extend_geometry(geometry)
        phi_size= geometry.phi_size
        ext_phi_size = extended_geometry.phi_size

        modes = torch.where(geometry.fourier_domain <= geometry.omega)[0].shape[0] #No padding used for fno (atm)
        sin2filler = FNO1d(modes, phi_size, ext_phi_size-phi_size, layer_widths=exp_fno_layers, verbose=True, dtype=torch.float).to(DEVICE)

        super().__init__(geometry, sin2filler, extended_geometry=extended_geometry, fbp=fbp, **kwargs)
    
    def extrapolate(self, X):
        N, phi_size, t_size = X.shape
        assert phi_size == self.geometry.phi_size
        filler = F.relu(self.sin2filler(X))
        assert filler.shape == (N, self.extended_geometry.phi_size-phi_size, t_size)

        return torch.concatenate([X, filler], dim=1) #full sinogram
    
    @classmethod
    def model_from_state_dict(clc, state_dict, final_fbp_class = RamLak, fbp_use_padding = True):
        ar, phi_size, t_size = state_dict['ar'], state_dict['phi_size'], state_dict['t_size']
        g = Geometry(ar, phi_size, t_size)
        #FIX change of attribute name sin2fill to sin2filler :()
        fixed_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("sin2fill."):
                fixed_state_dict["sin2filler."+k[9:]] = v
            else:
                fixed_state_dict[k] = v
        state_dict = fixed_state_dict

        fbp_sub_state_dict = {k[4:]: v for k, v in state_dict.items() if k.startswith("fbp.")}
        fbp = final_fbp_class.model_from_state_dict(fbp_sub_state_dict, use_padding=fbp_use_padding)

        fno_sub_state_dict = {k: v for k, v in state_dict.items() if k.startswith("")}
        exp_fno = fno_from_sd

        m = clc(g)
        m.load_state_dict(state_dict)
        return m


class CNNFiller(nn.Module):
    """
        Model that extrapolates sinogram using CNN blocks.
    """

    def __init__(self, from_height: int, to_height: int) -> None:
        super().__init__()

        self.from_height = from_height
        self.to_height = to_height
        self.n_blocks = ceil(to_height / from_height)
        self.convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(1, 64, (3,3), padding="same", dilation=1),
                nn.GELU(),
                nn.Conv2d(64, 64, (3,3), padding="same", dilation=3),
                nn.GELU(),
                nn.Conv2d(64, 64, (3,3), padding="same", dilation=3),
                nn.GELU(),
                nn.Conv2d(64, 1, (3,3), padding="same", dilation=1)
            ) for _ in range(self.n_blocks)
        ).to(DEVICE)
    
    def forward(self, X):
        return torch.concat([conv(X[:, None])[:, 0] for conv in self.convs], dim=1)[:, :self.to_height]

class CNNExtrapolatingBP(ExtrapolatingBP):
    def __init__(self, geometry: Geometry, fbp: ModelBase = None, **kwargs):

        extended_geometry = extend_geometry(geometry)
        super().__init__(geometry, CNNFiller(geometry.phi_size, extended_geometry.phi_size-geometry.phi_size), extended_geometry=extended_geometry, fbp=fbp)


class MomentFiller(nn.Module):

    def __init__(self, smp: SinoMoments, lr = 0.03, verbose = False, sino_mse_tol = 1e-4, max_iters = 2000) -> None:
        super().__init__()
        self.smp = smp
        self.lr = lr
        self.verbose = verbose
        self.sino_mse_tol = sino_mse_tol
        self.max_iters = max_iters
    
    def forward(self, X):
        N, Np, Nt = X.shape
        gap = self.smp.geometry.phi_size - Np
        pepper = torch.zeros(N, gap, Nt, dtype=X.dtype, device=DEVICE, requires_grad=True)
        old_pepper = torch.ones(N, gap, Nt, dtype=X.dtype, device=DEVICE)
        loptimizer = torch.optim.Adam([pepper], lr=0.03)
        iters = 0

        while torch.mean((pepper-old_pepper)**2) > self.sino_mse_tol and iters < self.max_iters:
            loptimizer.zero_grad()
            
            exp_sinos = torch.concat([X, pepper], dim=1)
            moms = [self.smp.get_moment(exp_sinos, ni) for ni in range(self.smp.n_moments)]
            proj_moms = [self.smp.project_moment(mom, ni) for ni, mom in enumerate(moms)]
            loss = sum(torch.mean((mom-p_mom)**2) for mom, p_mom in zip(moms, proj_moms)) / self.smp.n_moments

            loss.backward()
            loptimizer.step()

            iters += 1

        self.print_msg(f"sinos analytically extrapolated with moment diff {loss.item()} after iters {iters}")
        return pepper.detach()
    
    def print_msg(self, txt):
        if self.verbose:
            print(txt)



class MIFNO_BP(ExtrapolatingBP):

    def __init__(self, geometry: Geometry, n_moments = 12, sino_mse_tol = 1e-4, exp_max_iters = 2000, extended_geometry: Geometry = None,  use_padding=True, **kwargs):

        if extended_geometry == None: extended_geometry = extend_geometry(geometry)
        smp = SinoMoments(extended_geometry, n_moments=n_moments)
        sin2filler = MomentFiller(smp, verbose=True, sino_mse_tol=sino_mse_tol, max_iters=exp_max_iters)

        super().__init__(geometry, sin2filler, extended_geometry=extended_geometry, fbp="fno")


if __name__ == '__main__':
    geometry = Geometry(0.5, 160, 100, reco_shape=(256, 256))

    model = FNOExtrapolatingBP(geometry)

    train_sinos, train_y, test_sinos, test_y = setup(geometry, num_to_generate=0, train_ratio=0.9, use_realistic=True, data_path="data/kits_phantoms_256.pt")

    model.visualize_output(test_sinos, test_y, output_location="show")

        

