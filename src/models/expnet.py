import torch
import torch.nn as nn
import torch.nn.functional as F

import odl.contrib.torch as odl_torch

from utils.geometry import Geometry, extend_geometry, setup
from models.modelbase import ModelBase
from models.fbps import FBP
from models.analyticmodels import ramlak_filter
from utils.fno_1d import FNO1d



class ExtrapolationNet(ModelBase):

    def __init__(self, geometry: Geometry, **kwargs):
        super().__init__(geometry, **kwargs)

        self.extended_geometry = extend_geometry(geometry)

        phi_size, t_size = self.geometry.phi_size, self.geometry.t_size
        ext_phi_size, _ = self.extended_geometry.phi_size, self.extended_geometry.t_size

        assert ext_phi_size > phi_size

        # self.sin2fill = FNO1d(geometry.fourier_domain.shape[0], phi_size, ext_phi_size-phi_size, hidden_layer_widths=[40], verbose=True, dtype=torch.float)
        self.sin2fill = nn.Sequential(nn.Conv1d(phi_size, 10, kernel_size=1, padding="same", bias=False), nn.ReLU(), nn.Conv1d(10, ext_phi_size-phi_size, kernel_size=1, padding="same", bias=False), nn.ReLU())

        # self.sin2mom = nn.Conv2d(1, t_size, kernel_size=(1, t_size), padding="valid", bias=False)
        # self.mom2fillmom = nn.Conv1d(phi_size, ext_phi_size-phi_size, kernel_size=1, bias=False)
        # self.fillmom2fill = nn.Conv2d(1, t_size, kernel_size=(1, t_size), padding="valid", bias=False)

        self.fbp = FBP(self.extended_geometry, initial_kernel=ramlak_filter(self.extended_geometry), trainable_kernel=False)


    def forward(self, X: torch.Tensor):

        N, phi_size, t_size = X.shape
        ext_phi_size, ext_t_size = self.extended_geometry.phi_size, self.extended_geometry.t_size

        filler = self.sin2fill(X)

        # mom = F.relu(self.sin2mom(X[:, None])[:, :, :, 0]).permute(0,2,1)
        # fillmom = F.relu(self.mom2fillmom(mom))
        # filler = F.relu(self.fillmom2fill(fillmom[:, None])[:, :, :, 0]).permute(0,2,1)
        assert filler.shape == (N, ext_phi_size-phi_size, t_size)

        sinos = torch.concatenate([X, filler], dim=1) #full sinogram

        return self.fbp(sinos)


if __name__ == '__main__':
    geometry = Geometry(0.5, 160, 100, reco_shape=(256, 256))

    model = ExtrapolationNet(geometry)

    train_sinos, train_y, test_sinos, test_y = setup(geometry, num_to_generate=0, train_ratio=0.9, use_realistic=True, data_path="data/kits_phantoms_256.pt")

    model.visualize_output(test_sinos, test_y, output_location="show")

        

