"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 1D problem such as the (time-independent) Burgers equation discussed in Section 5.1 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).

Original author Zongyi Li, Modified to general layer widths by Emanuel Ström.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


################################################################################################################
#                                              2d fourier layer                                                #
################################################################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, max_mode_y, max_mode_x, dtype=None):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        Input: (batches, in_channels, dimension_y, dimension_x) - pytorch Tensor x

        output: (batches, out_channels, dimension) - pytorch Tensor y,
                determined by the relation:
                
                y = iFFT(  W  @  trunc(  FFT( x ) ) )
                
                iFFT, FFT are the 2D Fourier Transform and its inverse, respectively.
                trunc truncates the FFT of x to the lowest modes, determined by the max_mode_y and max_mode_x variables.
                W is a matrix with dimensions (in_channels, out_channels, max_mode_y, max_mode_x)
                @ is a matrix operation: (in_channels, (max_mode_y x max_mode_x)) -> (out_channels, (max_mode_y x max_mode_x))
                
                
                In function space (imagine input x as a function), the operation corresponds to
                
                y = K * x,
                
                where K is a (out_channel, in_channel) matrix of convolutional kernels,
                which operate on the channels i=1,2,3 ... n of x as follows
                
                y_i = k1i * x1  +  k2i * x2  +  k3i * x3  +  ...  +  kni * xn
                
                where * is the convolution operator.                
        """
        if (dtype == "float") or (dtype is torch.float):
            self.dtype = torch.float
            self.cdtype = torch.cfloat
        else:
            self.dtype = torch.double
            self.cdtype = torch.cdouble
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_y = max_mode_y  #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes_x = max_mode_x

        self.scale = (1 / (in_channels*out_channels))
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes_x, self.modes_y, dtype=self.cdtype))

    # Complex multiplication
    def compl_mul1d(self, inp, weights):
        # (batch, in_channel, y, x), (in_channel, out_channel, y, x) -> (batch, out_channel, y, x)
        return torch.einsum("biyx,ioyx->boyx", inp, weights)
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        # return torch.einsum("bix,iox->box", inp, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2+1,  device=x.device, dtype=self.cdtype)
        out_ft[:, :, :self.modes_y, :self.modes_x] = self.compl_mul1d(x_ft[:, :, :self.modes_y, :self.modes_x], self.weights)

        #Return to physical space
        x = torch.fft.irfft2(out_ft)
        return x

    
###############################################################################################################
#                                    FNO 2D Using spectral convolution module                                 #
###############################################################################################################
class FNO2d(nn.Module):
    def __init__(self, modes_y, modes_x, in_channels, out_channels, layer_widths=None, verbose=False, dtype=None):
        super(FNO2d, self).__init__()

        """
        The overall network. 
        1. Lift the input to the desired channel dimension by a channel wise linear operation (in_channels, layer_widths[0]).
        2. multiple layers of the integral operators u' = (W + K)(u),
           number of layers determined by the length of the vector layer_widths.
        3. Project from the channel space to the output space by another linear operation.
        
        
        input: A batch of vector valued functions x, represented as a discretised Tensor.
                
                functions:  a batch of b functions x: [0,1]^2 -> R^d ,  
                representation: Tensor             x in  R^(b * d * Ny*Nx), 

                where b is number of batches, d=in_channels, Ny=discretisation points along y-axis and Nx=discretisation points along x-axis.
                that is, x_ijkl is channel j of function i, evaluated at a point (x,y) = (t_l, s_k).

        output: a batch of vector valued functions y, represented same as input, but with d=out_channels.
        
        parameters: layer_widths regulates the number of channels in the inner layers
                    modes regulates the frequency cutoff in the SpectralConv1 operations.
        
        Function:
                depends on input as:
                
                yi = wi1 * zK1  +  wi2 * zK2  +  wi3 * zK3  +  ...  +  wiL * zKL,  where wij are weights, i=1,2,..., out_channels,
                                                                                   and where j = 1,2,..., layer_widths[-1].
                
                zK is the final layer of the inner network (K = len(layer_widths) is the number of layers),
                where the layers are updated according to 
                
                zk+1 = gelu(  W  @  zk     +    SpectralConv2d(   zk   )    +  bias),
                
                with zk a batched set of functions with dimension (b,  d,  Ny, Nx), (b functions [0,1] -> R^layer_widths[k])
                W the same type of linear operation as used in SpectralConv2d
                bias is a discretised function bias: [0,1]^2 -> R^layer_widths[k+1].
                gelu is the gelu activation function, operating element wise.
                
                The updates can be written in the continuum formulation like:
                
                z_{(k+1),i}(s) = gelu(  w_{ki1} * z_{k1}(s)  +  w_{ki2} * z_{k2}(s)  +  ...  +  w_{kin} * z_{kn}(s)        (Linear mixing)
                                      + (r_{ki1} x z_{k1})(s)  +  (r_{ki2} x z_{k2})(s)  +  ...  +  (r_{kin} x z_{kn})(s)  (Convolution)
                                       +  bias_i(s)  )                                                                     (bias)
                
                
                where n = layer_widths[k], and where i = 1,2, ... layer_widths[k+1].              
                
        """
        
        self.verbose = verbose
        
        if layer_widths is None:
            self.n_layers = 4
            self.layer_widths = [2 * in_channels,] * (self.n_layers+1)
            self.print_msg(f"Employing default layer structure, {self.layer_widths}")
        else:
            self.n_layers = len(layer_widths)-1
            self.layer_widths = layer_widths
            
        if (dtype == "float") or (dtype is torch.float):
            self.dtype = torch.float
            self.cdtype = torch.cfloat
        else:
            self.dtype = torch.double
            self.cdtype = torch.cdouble    
            
        
        self.inp = nn.Linear(in_channels, self.layer_widths[0], dtype=self.dtype)        
        
        # Convolution layers
        self.conv_list = nn.ModuleList([SpectralConv2d(self.layer_widths[i], self.layer_widths[i+1], modes_y, modes_x, dtype=self.dtype) for i in range(self.n_layers)])
        
        # Linear layers
        self.lin_list = nn.ModuleList([nn.Conv2d(self.layer_widths[i], self.layer_widths[i+1], (1,1), dtype=self.dtype) for i in range(self.n_layers)])
        

        self.out = nn.Linear(self.layer_widths[-1], out_channels, dtype=self.dtype)

        
    def print_msg(self, msg):
        if self.verbose:
            print(msg)
        pass

    def forward(self, x):
        
        # Project to FNO width
        #print(self.inp.weight.dtype)
        x = self.inp(x.permute(0,3,2,1)).permute(0,3,2,1)
        
        # Evaluate FNO
        for conv_op, lin_op in zip(self.conv_list, self.lin_list):
            x = F.gelu(conv_op(x) + lin_op(x))
        
        # Project to out_channels width
        return self.out(x.permute(0,3,2,1)).permute(0,3,2,1)

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=self.cdtype)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)