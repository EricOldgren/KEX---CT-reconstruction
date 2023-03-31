import torch
import torch.nn as nn
import odl.contrib.torch as odl_torch

from utils.geometry import Geometry
from utils.fno_1d import FNO1d
from models.analyticmodels import RamLak


from math import ceil




class CrazyKernels:

    def __init__(self, geometry: Geometry, angle_batch_size: int) -> None:
        #assert ceil(geometry.phi_size * 1.0 / geometry.ar) % angle_batch_size == 0, "bad choice"
        if angle_batch_size > 10: print("Big batch size unexpected may bahave unexpectedly")
        self.angle_batch_size = angle_batch_size
        
        self.geometry1 = geometry

        self.back1 = RamLak(geometry)

        full_phi_size  = round((geometry.phi_size * 1.0 / geometry.ar) / angle_batch_size) * angle_batch_size #maybe make cyclic in future
        self.geometry2 = Geometry(1.0, full_phi_size, geometry.t_size, reco_shape=geometry.reco_space.shape)
        modes = torch.where(self.geometry2.fourier_domain > self.geometry2.omega)[0].shape[0]
        self.ray_layer = odl_torch.OperatorModule(self.geometry2.ray)
        self.BP_layer = odl_torch.OperatorModule(self.geometry2.BP)

        self.fno = FNO1d(modes, in_channels=angle_batch_size, out_channels=angle_batch_size)

    def forward(self, sinos: torch.Tensor):
        N, phi_size, t_size = sinos.shape

        back_bad = self.back1(sinos)
        sinos_full: torch.Tensor = self.ray_layer(back_bad)

        sinos_full = sinos_full.view(-1, self.angle_batch_size, t_size)
        sinos_full = self.fno(sinos_full)
        sinos_full = sinos_full.view(N, phi_size, t_size)

        return self.BP_layer(sinos_full)