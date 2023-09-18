import torch
import torch.nn as nn

from abc import abstractmethod
from typing import Type

from utils.tools import DEVICE, DTYPE, PathType
from models.modelbase import FBPModelBase, _init_checkpoint_from_state_dict, _checkpoint_state_dict


def concat_pred_inp(inp: torch.Tensor, output: torch.Tensor, known_angles = None):
    res = torch.stack([inp, output], dim=-3)
    if known_angles is not None:
        return res[:,:, known_angles]
    return res

class DiscriminatorBase(torch.nn.Module):
    @abstractmethod
    def get_init_args(self):
        "get args used in init. required to save model."

class DCNN(DiscriminatorBase):

    def __init__(self, h: int, w: int, in_c: int, min_c: int = 8, max_c: int = 128):
        super().__init__()
        self._init_args = (h, w, in_c, min_c, max_c)
        
        self.inp_shape = (in_c, h, w)

        c = in_c
        next_c = lambda c : min_c if c == in_c else min(max_c, c*2)
        conv_layers = []
        while min(h, w) >= 4:
            conv_layers.append(nn.Conv2d(c, next_c(c), (4,4), 4, padding=0, device=DEVICE))
            c = next_c(c)
            h = h // 4
            w = w // 4
        
        conv_layers.append(nn.Conv2d(c, 1, (1,1), padding=0, device=DEVICE))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.ffnn = nn.Linear(h*w, 1, device=DEVICE)

    def get_init_args(self):
        return self._init_args

    def forward(self, inp: torch.Tensor):
        """
            Predict if input is real or generated data.

            Input: inp of shape N x in_c x h x w
            Output: probability of shape N x 1.
            Output is 1 if data is predicted as real and 0 if predicted as fake
        """
        assert not inp.isnan().any()
        N, c, h, w = inp.shape
        assert (c, h, w) == self.inp_shape, f"expected shape (..., {self.inp_shape})"
        out = inp
        for conv in self.conv_layers:
            assert not out.isnan().any()
            out = conv(out)
            out = nn.functional.leaky_relu(out, 0.2)
        
        return nn.functional.sigmoid(self.ffnn(out.reshape(N,-1)))
    
class Informed_DCNN(DiscriminatorBase):

    def __init__(self, h: int, w: int, in_c: int, n_conditions: int, min_c: int = 8, max_c: int = 128):
        super().__init__()
        self._init_args = (h, w, in_c, n_conditions, min_c, max_c)

        self.inp_shape = (in_c, h, w)
        self.n_conditions = n_conditions

        c = in_c
        next_c = lambda c : min_c if c == 1 else min(max_c, c*2)
        conv_layers = []
        while min(h, w) >= 4:
            conv_layers.append(nn.Conv2d(c, next_c(c), (4,4), 2, padding=1, device=DEVICE))
            c = next_c(c)
            h = h // 2
            w = w // 2
        
        conv_layers.append(nn.Conv2d(c, 1, (1,1), padding=0, device=DEVICE))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.ffnn = nn.Linear(h*w+n_conditions, 1)
    
    def get_init_args(self):
        return self._init_args

    def forward(self, inp: torch.Tensor, conditions: torch.Tensor):
        """
            Predict if input is real or generated data.

            Input: inp of shape N x in_c x h x w
            Conditions of shape N x n_conds
            Output: probability of shape N x 1.
            Output is 1 if data is predicted as real and 0 if predicted as fake
        """
        N, c, h, w = inp.shape
        assert (c, h, w) == self.inp_shape, f"expected shape (..., {self.inp_shape})"
        assert conditions.shape == (N, self.n_conditions)
        out = inp
        for conv in self.conv_layers:
            out = conv(inp)
            out = nn.functional.leaky_relu(out, 0.2)
        
        return nn.functional.sigmoid(self.ffnn(torch.concat([out.reshape(N,-1), conditions], dim=-1)))
    

def save_gan(G: FBPModelBase, D: DiscriminatorBase, optimizer, loss: torch.Tensor, ar: float, path: PathType):
    torch.save({
        "generator_checkpoint": _checkpoint_state_dict(G, optimizer, loss, ar),
        "discriminator_checkpoint": {
            "model_state_dict": D.state_dict(),
            "model_init_args": D.get_init_args()
        }
    }, path)
def load_gan(path: PathType, g_model: Type[FBPModelBase], d_model: Type[DiscriminatorBase]):
    state_dict = torch.load(path, map_location=DEVICE)
    g_checkpoint = _init_checkpoint_from_state_dict(state_dict["generator_checkpoint"], g_model)
    
    D = d_model(state_dict["discriminator_checkpoint"]["model_init_args"])
    D.load_state_dict(state_dict["discriminator_checkpoint"]["model_state_dict"])

    return g_checkpoint, D
