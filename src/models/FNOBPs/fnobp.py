import torch
import torch.nn as nn
import torch.nn.functional as F
from geometries import FBPGeometryBase, DEVICE, DTYPE
from models.modelbase import FBPModelBase, load_model_from_checkpoint
from utils.fno_1d import FNO1d

class FNO_BP(FBPModelBase):

    def __init__(self, geometry: FBPGeometryBase, hidden_layers: "list[int]", use_base_filter = True, modes = None) -> None:
        super().__init__()
        self._init_args = (hidden_layers, use_base_filter, modes)
        self.geometry = geometry

        if use_base_filter:
            self.basefilter = nn.Parameter(geometry.ram_lak_filter(), requires_grad=False)
        else:
            self.basefilter = nn.Parameter(geometry.ram_lak_filter()*0, requires_grad=False)
        if modes is None:
            modes = self.basefilter.nelement()
        self.fno1d = FNO1d(modes, geometry.n_projections, geometry.n_projections, hidden_layers, dtype=DTYPE).to(DEVICE)
    
    def get_init_torch_args(self):
        return self._init_args

    def get_extrapolated_sinos(self, sinos: torch.Tensor):
        "fno_bp does no extrapolation - sinos are returned"
        return sinos
    
    def get_extrapolated_filtered_sinos(self, sinos: torch.Tensor):
        return self.fno1d(sinos) + self.geometry.inverse_fourier_transform(self.geometry.fourier_transform(sinos*self.geometry.jacobian_det)*self.basefilter)

    def forward(self, sinos: torch.Tensor):
        return F.relu(self.geometry.project_backward(self.get_extrapolated_filtered_sinos(sinos)/2))
    
    @staticmethod
    def load(path):
        return load_model_from_checkpoint(path, FNO_BP)
                
        
