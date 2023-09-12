import torch
import torch.nn.functional as F

from typing import Tuple, List

from utils.tools import DEVICE, DTYPE, MSE
from utils.polynomials import POLYNOMIAL_FAMILY_MAP, Legendre
from utils.fno_1d import FNO1d
from geometries import FBPGeometryBase
from models.modelbase import FBPModelBase
from models.FNOBPs.fnobp import FNO_BP


class FNO_SinoExp(FBPModelBase):
    """
        FNO for generating sino
    """

    def __init__(self, geometry: FBPGeometryBase, ar: float, hidden_layers: List[int]) -> None:
        super().__init__()
        self.geometry = geometry
        self._init_args = (ar, hidden_layers)
        
        self.nin = self.geometry.n_known_projections(ar)
        self.nout = max(self.geometry.min_n_projs - self.nin, 0)
        if self.nout == 0:
            self.fno = None
        else:
            self.fno = FNO1d(geometry.projection_size//2, self.nin, self.nout, hidden_layers, dtype=DTYPE).to(DEVICE)
    
    def get_init_torch_args(self):
        return self._init_args

    def get_extrapolated_sinos(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out: torch.Tensor = None):

        N, _, _ = sinos.shape
        inp = sinos[:, known_angles]
        filler: torch.Tensor = self.fno(inp)
        assert not filler.isnan().any()
        filled_angles = torch.zeros(self.geometry.n_projections, dtype=torch.bool, device=DEVICE)
        filled_angles[:self.geometry.min_n_projs] = 1
        gap = self.geometry.n_projections - self.geometry.min_n_projs
        res = torch.concat([inp, filler, torch.zeros((N, gap, self.geometry.projection_size), device=DEVICE, dtype=DTYPE)], dim=1)
        self.geometry.reflect_fill_sinos(res, filled_angles)

        return res
    
    def get_extrapolated_filtered_sinos(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out: torch.Tensor = None):
        return self.geometry.inverse_fourier_transform(self.geometry.fourier_transform(self.get_extrapolated_sinos(sinos, known_angles)*self.geometry.jacobian_det)*self.geometry.ram_lak_filter()/2)
    
    def forward(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out: torch.Tensor = None):
        return self.geometry.project_backward(self.get_extrapolated_filtered_sinos(sinos, known_angles))
    
if __name__ == "__main__":
    from torch.utils.data import TensorDataset, DataLoader
    import matplotlib
    matplotlib.use("WebAgg")
    import matplotlib.pyplot as plt
    from geometries import HTC2022_GEOMETRY
    from utils.data import get_htc2022_train_phantoms, generate_htclike_batch
    from models.modelbase import save_model_checkpoint, plot_model_progress



    geometry = HTC2022_GEOMETRY
    PHANTOMS = torch.concat([get_htc2022_train_phantoms()[:-2], generate_htclike_batch(5, 5)])
    SINOS = geometry.project_forward(PHANTOMS)
    ar = 0.25

    model = FNO_SinoExp(geometry, ar, [40,40])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.8, 0.9))
    dataset = TensorDataset(PHANTOMS, SINOS)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

    n_epochs = 3000
    for epoch in range(n_epochs):

        for phantom_batch, sino_batch in dataloader:
            optimizer.zero_grad()

            sinos_la, known_angles = geometry.zero_cropp_sinos(sino_batch, ar, 0)

            exp = model.get_extrapolated_sinos(sinos_la, known_angles)

            loss = MSE(exp, sino_batch)
            assert not loss.isnan()
            loss.backward()

            optimizer.step()
            print("Epoch:", epoch+1, "loss:", loss.item())

    save_model_checkpoint(model, optimizer, loss, ar, "fno_sino_exp_v1.pt")
    plot_model_progress(model, SINOS, known_angles, None, PHANTOMS, 1, "FNO exp", True)

    

