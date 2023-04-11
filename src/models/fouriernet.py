import torch
import torch.nn as nn
import torch.nn.functional as F
import odl.contrib.torch as odl_torch

from utils.geometry import Geometry, DEVICE
from utils.fno_1d import FNO1d, SpectralConv1d
from models.analyticmodels import RamLak, ramlak_filter
from models.modelbase import ModelBase


from math import ceil
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import  random


class CrazyKernels(ModelBase):

    reconstructionfig: Figure = None

    def __init__(self, geometry: Geometry, angle_batch_size: int, layer_widths = []) -> None:
        super().__init__()

        if angle_batch_size > 10: print("Big batch size unexpected may bahave unexpectedly")
        self.angle_batch_size = angle_batch_size
        
        self.geometry1 = geometry

        self.back1 = RamLak(geometry)

        assert geometry.phi_size % self.angle_batch_size == 0
        modes1 = torch.where(self.geometry1.fourier_domain > self.geometry1.omega)[0].shape[0]
        self.spectralconv1 = SpectralConv1d(in_channels=angle_batch_size, out_channels=angle_batch_size, max_mode=modes1).to(DEVICE)
        self.add_module("spectralconv1", self.spectralconv1)
        self.BP_l1 = odl_torch.OperatorModule(self.geometry1.BP)

        full_phi_size  = round((geometry.phi_size * 1.0 / geometry.ar) / angle_batch_size) * angle_batch_size #maybe make cyclic in future
        self.geometry2 = Geometry(1.0, full_phi_size, geometry.t_size, reco_shape=geometry.reco_space.shape)
        modes = torch.where(self.geometry2.fourier_domain > self.geometry2.omega)[0].shape[0]
        self.ray_layer = odl_torch.OperatorModule(self.geometry2.ray)
        self.BP_layer = odl_torch.OperatorModule(self.geometry2.BP)

        self.fno = FNO1d(modes, in_channels=angle_batch_size, out_channels=angle_batch_size, layer_widths=layer_widths, dtype=torch.float).to(DEVICE)
        self.add_module("fno", self.fno)

    def forward(self, sinos: torch.Tensor):
        N, phi_size, t_size = sinos.shape

        filtered_bad = self.spectralconv1(sinos.view(-1, self.angle_batch_size, t_size)).view(N, phi_size, t_size)
        back_bad = self.BP_l1(filtered_bad)
        # back_bad = F.gelu(back_bad)
        #back_bad = self.back1(sinos)
        sinos_full: torch.Tensor = self.ray_layer(back_bad)
        sinos_full = F.gelu(sinos_full)

        sinos_full = self.fno(sinos_full.view(-1, self.angle_batch_size, t_size)).view(N, self.geometry2.phi_size, t_size)
        # sinos_full = F.gelu(sinos_full)

        return self.BP_layer(sinos_full)