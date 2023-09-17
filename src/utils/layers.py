import torch
import torch.nn.functional as F
from math import sqrt
import numpy as np

from utils.tools import DEVICE, DTYPE

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    d = Q.shape[-1]
    return F.softmax(Q@K.T / sqrt(d), dim=-1)@V

# Code from https://www.tensorflow.org/tutorials/text/transformer
def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, None],
                          np.arange(d_model)[None, :],
                          d_model)
  
  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
  pos_encoding = angle_rads[None, ...]
    
  return torch.from_numpy(pos_encoding).to(device=DEVICE, dtype=DTYPE)

class MultiHeadAttention(torch.nn.Module):
    "MultiHeaded attention"

    def __init__(self, qin_dim: int, kin_dim: int, vin_dim: int, nheads = 8, dk = 512, dout = 512):
        super().__init__()
        assert dk % nheads == 0
        self.nheads = nheads
        self.dk = dk
        self.head_dim = dk // nheads

        self.Wqs = torch.nn.Parameter(torch.randn((nheads, qin_dim, self.head_dim), device=DEVICE, dtype=DTYPE)/(self.head_dim*qin_dim), requires_grad=True)
        self.Wks = torch.nn.Parameter(torch.randn((nheads, kin_dim, self.head_dim), device=DEVICE, dtype=DTYPE)/(self.head_dim*kin_dim), requires_grad=True)
        self.Wvs = torch.nn.Parameter(torch.randn((nheads, vin_dim, self.head_dim), device=DEVICE, dtype=DTYPE)/(self.head_dim*vin_dim), requires_grad=True)
        self.Wout = torch.nn.Parameter(torch.randn((dk, dout), device=DEVICE, dtype=DTYPE)/(dk*dout), requires_grad=True) 

    def forward(self, qin: torch.Tensor, kin: torch.Tensor, vin: torch.Tensor):
        """
            shapes: (...) must coincide
                qin: ... x nqueries x qin_dim

                kin: ... x nvals x kin_dim
                
                vin: ... x nvals x vin

                return shape: ... x nqueries x dout
        """
        Qs = torch.einsum("...qi,hio->...hqo", qin, self.Wqs)
        Ks = torch.einsum("...ki,hio->hko", kin, self.Wks)
        Vs = torch.einsum("...vi,hio->hvo", vin, self.Wvs)

        return torch.concat([
            scaled_dot_product_attention(Qs[...,hi,:,:], Ks[...,hi,:,:], Vs[...,hi,:,:]) for hi in range(self.nheads)
        ], dim=-1)@self.Wout