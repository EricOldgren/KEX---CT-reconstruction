from utils.geometry import BasicModel, Geometry, DEVICE
from odl.contrib import torch as odl_torch
import torch
import numpy as np
from math import ceil

class RamLak(BasicModel):

    def __init__(self, geometry: Geometry, **kwargs):
        "FBP based on Radon's invesrion formula and Nattarer's sampling theorem. Uses a |x| filter with a hard cut"
        super(RamLak, self).__init__(geometry,**kwargs)
        
        self.geometry = geometry
        self.BP_layer = odl_torch.OperatorModule(geometry.BP)

        self.kernel = torch.nn.Parameter(geometry.fourier_domain /  (2*np.pi), requires_grad=False)
        self.kernel[geometry.fourier_domain > geometry.omega] = 0
