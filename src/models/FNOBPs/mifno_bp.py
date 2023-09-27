import torch
import torch.nn.functional as F

from typing import Tuple, List

from utils.tools import DEVICE, DTYPE, MSE
from utils.polynomials import POLYNOMIAL_FAMILY_MAP, Legendre
from utils.fno_1d import FNO1d
from geometries import FBPGeometryBase
from models.modelbase import FBPModelBase
from models.FNOBPs.fnobp import FNO_BP_orig


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
    from utils.tools import MSE, htc_score
    from utils.polynomials import Legendre, Chebyshev
    from src.geometries.data import get_htc2022_train_phantoms, get_htc_trainval_phantoms, GIT_ROOT
    from geometries import HTC2022_GEOMETRY
    from models.modelbase import plot_model_progress, save_model_checkpoint
    import matplotlib
    import matplotlib.pyplot as plt
    from statistics import mean
    ar = 0.25
    geometry = HTC2022_GEOMETRY
    PHANTOMS, VALIDATION_PHANTOMS = get_htc_trainval_phantoms()
    # PHANTOMS = VALIDATION_PHANTOMS = get_htc2022_train_phantoms()
    print("phantoms are loaded")
    SINOS = geometry.project_forward(PHANTOMS)
    print("sinos are calculated")
    dataset = TensorDataset(PHANTOMS, SINOS)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    M, K = 128, 64

    model = FNO_SinoExp(geometry, ar, [100, 100, 100])
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
    warmup_steps = 50
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : min((epoch+1)**-0.5, (epoch+1)*warmup_steps**-1.5))

    n_epochs = 300
    for epoch in range(n_epochs):
        sino_losses, recon_losses = [], []
        for phantom_batch, sino_batch in dataloader:
            optimizer.zero_grad()

            la_sinos, known_angles = geometry.zero_cropp_sinos(sino_batch, ar, 0)

            exp_sinos = model.get_extrapolated_sinos(la_sinos, known_angles)
            mse_sinos = MSE(exp_sinos, sino_batch)

            mse_sinos.backward()
            optimizer.step()
            sino_losses.append(mse_sinos.item())

        scheduler.step()
        print("epoch:", epoch, "sino loss:", mean(sino_losses))

    VALIDATION_SINOS = geometry.project_forward(VALIDATION_PHANTOMS)
    _, known_angles = geometry.zero_cropp_sinos(VALIDATION_SINOS, ar, 0)

    disp_ind = 1
    save_model_checkpoint(model, optimizer, mse_sinos, ar, GIT_ROOT / f"data/models/fno_expv1_sino_mse_{mean(sino_losses)}.pt")
    plot_model_progress(model, VALIDATION_SINOS, known_angles, VALIDATION_PHANTOMS, disp_ind=disp_ind)
    
    for i in plt.get_fignums():
        fig = plt.figure(i)
        title = fig._suptitle.get_text() if fig._suptitle is not None else f"fig{i}"
        plt.savefig(f"{title}.png")