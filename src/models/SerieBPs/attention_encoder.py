import torch
import torch.nn.functional as F


from utils.polynomials import Legendre, POLYNOMIAL_FAMILY_MAP
from geometries import FBPGeometryBase
from models.modelbase import FBPModelBase


MODEL_DIM = 512

class AttentionEncoder(FBPModelBase):

    def __init__(self, geometry: FBPGeometryBase, ar: float):
        super().__init__()
        self._init_args = (geometry, ar)
        self.geometry = geometry
        self.ar = ar

    def get_init_torch_args(self):
        return self._init_args
