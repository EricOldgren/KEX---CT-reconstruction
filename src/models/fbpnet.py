import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from utils.geometry import Geometry, BasicModel, DEVICE
from models.modelbase import ModelBase
from models.fbps import FBP, SmoothFBP

from typing import Literal
from typing import Mapping, Any


class FBPNet(ModelBase):

    def __init__(self, geometry: Geometry, n_fbps = 8, modelblock = FBP, use_bias=True, **kwargs):
        "2 layer network consisting of sums of FBPs"

        super(FBPNet, self).__init__(geometry, **kwargs)
        self.plotkernels = True

        self.fbps = [modelblock(geometry) for _ in range(n_fbps)]

        self.weights = nn.Parameter(torch.randn(n_fbps).to(DEVICE))
        if use_bias == True: self.bias = nn.Parameter(torch.randn(n_fbps).to(DEVICE))
        else: self.bias = nn.Parameter(torch.zeros(n_fbps).to(DEVICE), requires_grad=False)
        self.relu = nn.ReLU()

        for i, fbp in enumerate(self.fbps):
            self.add_module(f"fbp{i}", fbp)

    def kernels(self):
        return [fbp.kernel for fbp in self.fbps]

    def forward(self, x):
        constructs = torch.stack([self.relu(fbp(x) + b) for fbp, b in zip(self.fbps, self.bias)])

        out = torch.sum(constructs*self.weights[:, None, None, None], axis=0)

        return self.relu(out)

    def regularization_term(self):
        "Returns a sum which penalizies large kernel values at large frequencies, in accordance with Nattarer's sampling Theorem"
        
        return sum(self.weights[i]*self.weights[i]*self.fbps[i][0].regularisation_term() for i in range(len(self.fbps)) ) / len(self.fbps)


def load_fbpnet_from_dict(path, smooth_filters: bool = False):
    sd = torch.load(path)
    ar, phi_size, t_size = sd["ar"], sd["phi_size"], sd["t_size"]
    geometry = Geometry(ar, phi_size, t_size)
    ret = FBPNet(geometry, n_fbps=sd["n_fbps"], use_smooth_filters=smooth_filters)
    ret.load_state_dict(sd)
    return ret

