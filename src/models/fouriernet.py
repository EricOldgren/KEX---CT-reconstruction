import torch
import torch.nn as nn
import torch.nn.functional as F
import odl.contrib.torch as odl_torch

from utils.geometry import Geometry, DEVICE, extend_geometry
from utils.fno_1d import FNO1d, SpectralConv1d
from models.analyticmodels import RamLak, ramlak_filter
from models.modelbase import ModelBase

from typing import Literal
from math import ceil
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import  random


class FNO_BP(ModelBase):

    def __init__(self, geometry: Geometry, angle_batch_size: int, layer_widths = [], dtype=torch.float64, use_basefliter = True, basefilter: 'torch.Tensor|None' = None, trainable_basefilter = False, **kwargs) -> None:
        """
            Back projection model filtering with a combination of an FNO module and a kernel.

            returns: Relu( BP( FNO(X) + basefilter * X) ), where '*' denotes convolution

            Args:
                - geometry (Geometry): the geometry
                - angle_batch_size (int): number of rows of a sinogram to convolve together throughth the fno.
                - layer_widths (list[int]): hidden layer sizes of the fno
                - dtype (torch.dtype): data type - NOTE a real dtype should be used, fno automatically creates a complex version of it
                - use_basfilter (bool): if set to False the basefilter will be constant zero
                - basefilter (Tensor): initial basefilter to use given in fourier domain, if set to None the ramlak filter is used
                - trainable_basefilter (bool): wether the basefilter is tracked with autograd 
        """
        super().__init__(geometry, **kwargs)
        self.plotkernels = True

        self.angle_batch_size = angle_batch_size
        
        assert geometry.phi_size % self.angle_batch_size == 0, "phi_size and angle batch size must match"
        modes = torch.where(geometry.fourier_domain <= geometry.omega)[0].shape[0]
        
        self.fno = FNO1d(modes, in_channels=angle_batch_size, out_channels=angle_batch_size, hidden_layer_widths=layer_widths, dtype=dtype, verbose=True).to(DEVICE)
        self.add_module("fno", self.fno)

        #Init basefilter
        if not use_basefliter:
            self.basefilter = nn.Parameter(torch.zeros(geometry.fourier_domain.shape, device=DEVICE, dtype=dtype), requires_grad=False)
        else:
            cdtype = torch.cfloat if dtype == torch.float else torch.cdouble #complex dtype
            if basefilter == None:
                self.basefilter = nn.Parameter(ramlak_filter(geometry, cdtype), requires_grad=trainable_basefilter) #default to ramlak
            else:
                assert basefilter.shape == geometry.fourier_domain.shape, "wrong formatted basefilter"
                self.basefilter = nn.Parameter(basefilter.to(DEVICE, dtype=cdtype), requires_grad=trainable_basefilter)
    
    def kernels(self) -> 'list[torch.Tensor]':
        return [self.basefilter]

    def forward(self, X: torch.Tensor):
        N, phi_size, t_size = X.shape
        assert phi_size % self.angle_batch_size == 0, f"this shape {X.shape} is incompatible with the given fno."
 
        out = self.fno(X.view(-1, self.angle_batch_size, t_size)).view(N, phi_size, t_size)

        out = out + self.geometry.inverse_fourier_transform(self.geometry.fourier_transform(X) * self.basefilter)

        return F.relu(self.BP_layer(out))
    
class GeneralizedFNO_BP(ModelBase):
    
    def __init__(self, geometry: Geometry, fno: nn.Module, extended_geometry: Geometry = None, dtype=torch.float32, use_basefliter = True, basefilter: 'torch.Tensor|None' = None, trainable_basefilter = False, **kwargs) -> None:
        """
            Back projection model filtering with a combination of an FNO module and a kernel.

            returns: Relu( BP( fno(X) + ZERO_PAD(basefilter * X)) ), where '*' denotes convolution

            Args:
                - geometry (Geometry): the geometry
                - fno (Module): module that maps a sinogram from the given geometry to a filtered sinogram of the extended geometry
                - etended_geometry (Geometry): geometry to perform  backprojection from - fno should map sinograms from geometry to extended_geometry
                - dtype (torch.dtype): data type of basefilter.real - should be float32 or float64
                - use_basfilter (bool): if set to False the basefilter will be constant zero
                - basefilter (Tensor): initial basefilter to use given in fourier domain, if set to None the ramlak filter is used
                - trainable_basefilter (bool): wether the basefilter is tracked with autograd 
        """
        super().__init__(geometry, **kwargs)
        self.plotkernels = True
        
        cdtype = torch.cfloat if dtype == torch.float else torch.cdouble #complex dtype
        
        self.fno = fno.to(DEVICE)
        self.add_module("fno", self.fno)

        if extended_geometry == None:
            self.extended_geometry = extend_geometry(geometry)
        else:
            assert extended_geometry.phi_size >= geometry.phi_size, "extended geometry should contain geometry as a subgeometry"
            self.extended_geometry = extended_geometry
        self.extended_BP_layer = odl_torch.OperatorModule(self.extended_geometry.BP)

        #Init basefilter
        if not use_basefliter:
            self.basefilter = nn.Parameter(torch.zeros(geometry.fourier_domain.shape, device=DEVICE, dtype=dtype), requires_grad=False)
        else:
            if basefilter == None:
                self.basefilter = nn.Parameter(ramlak_filter(geometry, cdtype), requires_grad=trainable_basefilter) #default to ramlak
            else:
                assert basefilter.shape == geometry.fourier_domain.shape, "wrong formatted basefilter"
                self.basefilter = nn.Parameter(basefilter.to(DEVICE, dtype=cdtype), requires_grad=trainable_basefilter)
    
    def convert(self, geometry: Geometry):
        "Not convertible."
        raise NotImplementedError("Converting this model has not been impleemneted.")

    def kernels(self) -> 'list[torch.Tensor]':
        return [self.basefilter]

    def forward(self, X: torch.Tensor):
        N, phi_size, t_size = X.shape

        out = self.fno(X)
        #print(out.shape)
        #print(N, self.extended_geometry.phi_size, self.extended_geometry.t_size)
        assert out.shape == (N, self.extended_geometry.phi_size, self.extended_geometry.t_size), "fno incompatible with geometries"

        out_base = self.geometry.inverse_fourier_transform(self.geometry.fourier_transform(X) * self.basefilter)
        unknown = torch.zeros(N, self.extended_geometry.phi_size - phi_size, t_size, device=DEVICE)

        out = out + torch.concatenate([out_base, unknown], dim=1)

        return F.relu(self.extended_BP_layer(out))
    
    def return_sino(self, X: torch.Tensor):
        N, phi_size, t_size = X.shape

        out = self.fno(X)
        #print(out.shape)
        #print(N, self.extended_geometry.phi_size, self.extended_geometry.t_size)
        assert out.shape == (N, self.extended_geometry.phi_size, self.extended_geometry.t_size), "fno incompatible with geometries"

        out_base = self.geometry.inverse_fourier_transform(self.geometry.fourier_transform(X) * self.basefilter)
        unknown = torch.zeros(N, self.extended_geometry.phi_size - phi_size, t_size, device=DEVICE)

        out = out + torch.concatenate([out_base, unknown], dim=1)

        return out
    
    @classmethod
    def model_from_state_dict(clc, state_dict):
        ar, phi_size, t_size = state_dict['ar'], state_dict['phi_size'], state_dict['t_size']
        g = Geometry(ar, phi_size, t_size)
        
        fno_sd = {k[4:]: v for k, v in state_dict.items() if k.startswith("fno.")}
        in_channels, out_channels, modes = fno_sd["conv_list.0.weights"].shape
        dtype = torch.float if state_dict["basefilter"].dtype == torch.cfloat else torch.double
        hidden_layer_widths = []
        li = 1
        while True:
            if f"conv_list.{li}.weights" in fno_sd:
                w = fno_sd[f"conv_list.{li}.weights"]
                hidden_layer_widths.append(w.shape[0])
                out_channels = w.shape[1]
                li += 1
            else:
                break

        fno = FNO1d(modes, in_channels, out_channels, hidden_layer_widths=hidden_layer_widths, verbose=True, dtype=dtype)

        m = clc(g, fno, dtype=dtype)
        m.load_state_dict(state_dict)

        return m




