import torch
import torch.nn as nn
import torch.nn.functional as F
from geometries import FBPGeometryBase, DEVICE, DTYPE
from models.modelbase import FBPModelBase, load_model_checkpoint
from utils.fno_1d import FNO1d

class FNO_BP(FBPModelBase):

    def __init__(self, geometry: FBPGeometryBase, hidden_layers: "list[int]", n_known_angles: int, n_angles_out: int, use_base_filter = True, modes = None) -> None:
        super().__init__()
        self._init_args = (hidden_layers, n_known_angles, n_angles_out, use_base_filter, modes)
        self.geometry = geometry
        self.n_known_angles, self.n_angles_out = n_known_angles, n_angles_out

        if use_base_filter:
            self.basefilter = nn.Parameter(geometry.ram_lak_filter(), requires_grad=False)
        else:
            self.basefilter = nn.Parameter(geometry.ram_lak_filter()*0, requires_grad=False)
        if modes is None:
            modes = self.geometry.projection_size // 2
        self.fno1d = FNO1d(modes, n_known_angles, n_angles_out, hidden_layers, dtype=DTYPE).to(DEVICE)
    
    def get_init_torch_args(self):
        return self._init_args

    def get_extrapolated_sinos(self, sinos: torch.Tensor, *args):
        "fno_bp does no extrapolation - sinos are returned"
        return sinos
    
    def get_extrapolated_filtered_sinos(self, sinos: torch.Tensor, known_beta_bool: torch.Tensor, out_beta_bool: torch.Tensor):
        inp = sinos[:, known_beta_bool]
        res = sinos *0
        res[:, known_beta_bool] += self.geometry.inverse_fourier_transform(self.geometry.fourier_transform(inp*self.geometry.jacobian_det)*self.basefilter)
        res[:, out_beta_bool] += self.fno1d(inp)
        self.geometry.reflect_fill_sinos(res, out_beta_bool)
        return res

    def forward(self, sinos: torch.Tensor, knwon_beta_bool: torch.Tensor, out_beta_bool: torch.Tensor):
        return F.relu(self.geometry.project_backward(self.get_extrapolated_filtered_sinos(sinos, knwon_beta_bool, out_beta_bool)/2))
    
    @staticmethod
    def load_checkpoint(path):
        return load_model_checkpoint(path, FNO_BP)
                
        
