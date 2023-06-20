import torch
import torch.nn as nn
import torch.nn.functional as F
import odl.contrib.torch as odl_torch

from utils.geometry import Geometry, DEVICE, extend_geometry
from utils.modified_fno1d import FNO1d as moded_FNO1d
from utils.fno_1d import FNO1d, SpectralConv1d
from models.analyticmodels import RamLak, ramlak_filter
from models.modelbase import ModelBase

from typing import Literal
from math import ceil
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import random


class FNO_BP(ModelBase):

    def __init__(self, geometry: Geometry, angle_batch_size: int, layer_widths = [10,10], dtype=torch.float64, use_basefliter = True, basefilter: 'torch.Tensor|None' = None, use_padding = True, trainable_basefilter = False, **kwargs) -> None:
        """
            Back projection model filtering with a combination of an FNO module and a kernel.

            returns: Relu( BP( FNO(X) + basefilter * X) ), where '*' denotes convolution

            Args:
                - geometry (Geometry): the geometry
                - angle_batch_size (int): number of rows of a sinogram to convolve together at a time throughth the fno.
                - layer_widths (list[int]): layer sizes of the fno
                - dtype (torch.dtype): data type - NOTE a real dtype should be used, fno automatically creates a complex version of it
                - use_basfilter (bool): if set to False the basefilter will be constant zero
                - basefilter (Tensor): initial basefilter to use given in fourier domain, if set to None the ramlak filter is used
                - trainable_basefilter (bool): wether the basefilter is tracked with autograd 
        """
        super().__init__(geometry, **kwargs)
        self.plotkernels = True
        self.use_padding = use_padding

        self.angle_batch_size = angle_batch_size
        
        assert geometry.phi_size % self.angle_batch_size == 0, "phi_size and angle batch size must match"
        omgs = geometry.fourier_domain_padded if use_padding else geometry.fourier_domain
        modes = torch.where(geometry.fourier_domain <= geometry.omega)[0].shape[0] #No padding used for fno (atm)
        
        self.fno = FNO1d(modes, in_channels=angle_batch_size, out_channels=angle_batch_size, layer_widths=layer_widths, dtype=dtype, verbose=True).to(DEVICE)
        self.add_module("fno", self.fno)

        #Init basefilter
        if not use_basefliter:
            self.basefilter = nn.Parameter(torch.zeros(omgs.shape, device=DEVICE, dtype=dtype), requires_grad=False)
        else:
            cdtype = torch.cfloat if dtype == torch.float else torch.cdouble #complex dtype
            if basefilter == None:
                self.basefilter = nn.Parameter(ramlak_filter(geometry, padding=use_padding, dtype=cdtype), requires_grad=trainable_basefilter) #default to ramlak
            else:
                assert basefilter.shape == omgs.shape, "wrong formatted basefilter"
                self.basefilter = nn.Parameter(basefilter.to(DEVICE, dtype=cdtype), requires_grad=trainable_basefilter)
    
    def kernels(self) -> 'list[torch.Tensor]':
        return [self.basefilter]

    def forward(self, X: torch.Tensor):
        N, phi_size, t_size = X.shape
        assert phi_size % self.angle_batch_size == 0, f"this shape {X.shape} is incompatible with the given fno."
 
        out = self.fno(X.reshape(-1, self.angle_batch_size, t_size)).reshape(N, phi_size, t_size)

        out = out + self.geometry.inverse_fourier_transform(self.geometry.fourier_transform(X, padding=self.use_padding) * self.basefilter, padding=self.use_padding)

        return F.relu(self.BP_layer(out))
    
class GeneralizedFNO_BP(ModelBase):
    
    def __init__(self, geometry: Geometry, fno: nn.Module, extended_geometry: Geometry = None, dtype=torch.float32, use_basefliter = True, basefilter: 'torch.Tensor|None' = None, use_padding = True, trainable_basefilter = False, **kwargs) -> None:
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
        self.use_padding = use_padding
        
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
        omgs = geometry.fourier_domain_padded if use_padding else geometry.fourier_domain
        if not use_basefliter:
            self.basefilter = nn.Parameter(torch.zeros(omgs.shape, device=DEVICE, dtype=dtype), requires_grad=False)
        else:
            if basefilter == None:
                self.basefilter = nn.Parameter(ramlak_filter(geometry, padding=use_padding, dtype=cdtype), requires_grad=trainable_basefilter) #default to ramlak
            else:
                assert basefilter.shape == omgs.shape, "wrong formatted basefilter"
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

        out_base = self.geometry.inverse_fourier_transform(self.geometry.fourier_transform(X, padding=self.use_padding) * self.basefilter, padding=self.use_padding)
        unknown = torch.zeros(N, self.extended_geometry.phi_size - phi_size, t_size, device=DEVICE)

        out = out + torch.concatenate([out_base, unknown], dim=1)

        return F.relu(self.extended_BP_layer(out))
    
    def return_sino(self, X: torch.Tensor):
        N, phi_size, t_size = X.shape

        out = self.fno(X)
        #print(out.shape)
        #print(N, self.extended_geometry.phi_size, self.extended_geometry.t_size)
        assert out.shape == (N, self.extended_geometry.phi_size, self.extended_geometry.t_size), "fno incompatible with geometries"

        out_base = self.geometry.inverse_fourier_transform(self.geometry.fourier_transform(X, padding=self.use_padding) * self.basefilter, padding=self.use_padding)
        unknown = torch.zeros(N, self.extended_geometry.phi_size - phi_size, t_size, device=DEVICE)

        out = out + torch.concatenate([out_base, unknown], dim=1)

        return out
    
    def return_fno_sino(self,X: torch.Tensor):
        N, phi_size, t_size = X.shape

        out = self.fno(X)
        #print(out.shape)
        #print(N, self.extended_geometry.phi_size, self.extended_geometry.t_size)
        assert out.shape == (N, self.extended_geometry.phi_size, self.extended_geometry.t_size), "fno incompatible with geometries"

        out_base = self.geometry.inverse_fourier_transform(self.geometry.fourier_transform(X, padding=self.use_padding) * self.basefilter, padding=self.use_padding)
        unknown = torch.zeros(N, self.extended_geometry.phi_size - phi_size, t_size, device=DEVICE)

        return out
    
    @classmethod
    def model_from_state_dict(clc, state_dict, use_padding = False):
        ar, phi_size, t_size = state_dict['ar'], state_dict['phi_size'], state_dict['t_size']
        g = Geometry(ar, phi_size, t_size)
        
        dtype = torch.float if state_dict["basefilter"].dtype == torch.cfloat else torch.double
    
        fno_sd = {k[4:]: v for k, v in state_dict.items() if k.startswith("fno.")}
        fno = fno_from_sd(fno_sd, dtype=dtype)

        m = clc(g, fno, dtype=dtype, use_padding=use_padding)
        m.load_state_dict(state_dict)

        return m
    
def fno_from_sd(state_dict, dtype = None):
    _, in_channels = state_dict["inp.weight"].shape
    out_channels, _ = state_dict["out.weight"].shape

    layer_widths = []
    modes = 0

    li = 0
    while True:
        if f"conv_list.{li}.weights" in state_dict:
            w = state_dict[f"conv_list.{li}.weights"]
            if li == 0:
                layer_widths.extend([w.shape[0], w.shape[1]])
            else:
                assert w.shape[0] == layer_widths[-1]
                layer_widths.append(w.shape[1])
            modes = w.shape[-1]
            li += 1
        else:
            break
    if len(layer_widths): assert state_dict["inp.weight"].shape[0] == layer_widths[0] and state_dict["out.weight"].shape[1] == layer_widths[-1]

    return FNO1d(modes, in_channels, out_channels, layer_widths=layer_widths, verbose=True, dtype=dtype).to(DEVICE)

def moded_fno_from_sd(state_dict, dtype=None):

    in_channels, out_channels, modes = state_dict["conv_list.0.weights"].shape
    hidden_layer_widths = []
    li = 1
    while True:
        if f"conv_list.{li}.weights" in state_dict:
            w = state_dict[f"conv_list.{li}.weights"]
            hidden_layer_widths.append(w.shape[0])
            out_channels = w.shape[1]
            li += 1
        else:
            break

    return moded_FNO1d(modes, in_channels, out_channels, hidden_layer_widths=hidden_layer_widths, verbose=True, dtype=dtype).to(DEVICE)




if __name__ == '__main__':

    fno = FNO1d(100, 20, 20, [20,20])

    print("hello")


