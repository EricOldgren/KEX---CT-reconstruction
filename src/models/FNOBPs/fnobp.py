import torch
import torch.nn as nn
import torch.nn.functional as F
from geometries import FBPGeometryBase, DEVICE, DTYPE, naive_sino_filling
from geometries.extrapolation import RidgeSinoFiller
from models.modelbase import FBPModelBase, load_model_checkpoint
from utils.polynomials import Chebyshev, POLYNOMIAL_FAMILY_MAP
from utils.fno_1d import FNO1d
from typing import Any, List

class FNO_BP(FBPModelBase):

    def __init__(self, geometry: FBPGeometryBase, ar: float, hidden_layers: List[int], use_base_filter = True, modes = None, M = 100, K = 100, PolynomialFamilyKey: int = Chebyshev.key, l2_reg = 0.01):
        super().__init__()
        self._init_args = (ar, hidden_layers, use_base_filter, modes, M, K, PolynomialFamilyKey, l2_reg)
        self.geometry = geometry
        self.ar = ar
        self.M, self.K = M, K
        self.PolynomialFamily = POLYNOMIAL_FAMILY_MAP[PolynomialFamilyKey]
        self.l2_reg = torch.nn.Parameter(torch.tensor(l2_reg), requires_grad=False)

        if use_base_filter:
            self.basefilter = nn.Parameter(geometry.ram_lak_filter(), requires_grad=False)
        else:
            self.basefilter = nn.Parameter(geometry.ram_lak_filter()*0, requires_grad=False)
        if modes is None:
            modes = self.geometry.projection_size // 2

        self.known_angles = torch.zeros(geometry.n_projections, device=DEVICE, dtype=torch.bool)
        self.known_angles[:geometry.n_known_projections(ar)] = 1

        self.sinofiller = RidgeSinoFiller(geometry, self.known_angles, M, K, self.PolynomialFamily)
        self.fno1d = FNO1d(modes, geometry.n_projections, geometry.n_projections, hidden_layers, dtype=DTYPE).to(DEVICE)
    
    def get_init_torch_args(self):
        return self._init_args
    
    def get_extrapolated_sinos(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out = None):
        assert (known_angles == self.known_angles).all(), "rotate sinos so that first known angle is at index 0 to make inference with this model"
        exp = self.sinofiller.forward(sinos, self.l2_reg)
        reflected, known_reg = self.geometry.reflect_fill_sinos(sinos+0, self.known_angles)
        reflected[:, ~known_reg] = exp[:, ~known_reg]
        return reflected
    
    def get_extrapolated_filtered_sinos(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out = None):
        sinos = self.get_extrapolated_sinos(sinos, known_angles)
        return self.geometry.inverse_fourier_transform(self.geometry.fourier_transform(sinos*self.geometry.jacobian_det)*self.geometry.ram_lak_filter()/2) + self.fno1d(sinos)

    def forward(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out = None, use_relu = True):
        if use_relu:
            return F.relu(self.geometry.project_backward(self.get_extrapolated_filtered_sinos(sinos, known_angles)))
        return self.geometry.project_backward(self.get_extrapolated_filtered_sinos(sinos, known_angles))


class FNO_BP_orig(FBPModelBase):

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
        return load_model_checkpoint(path, FNO_BP_orig)
                
        
if __name__ == "__main__":
    from utils.tools import MSE, GIT_ROOT
    from geometries.data import HTC2022_GEOMETRY, get_synthetic_htc_phantoms, get_htc_traindata
    from models.modelbase import save_model_checkpoint, plot_model_progress
    from torch.utils.data import TensorDataset, DataLoader
    from statistics import mean
    import time
    import matplotlib.pyplot as plt
    import random
    import sys

    ar = int(sys.argv[1]) / HTC2022_GEOMETRY.n_projections
    hidden_layers = list(map(int, sys.argv[2:]))
    print("Training for ar:", ar)
    print("Using hidden layers:", hidden_layers)

    geometry = HTC2022_GEOMETRY
    print("loading phantoms...")
    PHANTOMS = get_synthetic_htc_phantoms(use_kits=False)
    print("phantoms loaded:", PHANTOMS.shape, PHANTOMS.dtype)
    print("calculating sinos...")
    SINOS = geometry.project_forward(PHANTOMS)
    VAL_SINOS, VAL_PHANTOMS = get_htc_traindata()
    M, K, ridge_reg = 50, 50, 0.01
    relu_in_training = False

    model = FNO_BP(geometry, ar, hidden_layers, M=M,K=K, PolynomialFamilyKey=Chebyshev.key, l2_reg=ridge_reg)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, weight_decay=3e-7)
    n_epochs = 40

    dataset = TensorDataset(PHANTOMS, SINOS)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    best_valloss = 1.0

    for epoch in range(n_epochs):
        if ((epoch)%5) == 0:
            print("Validation results:")
            val_loss = plot_model_progress(model, VAL_SINOS, ar, VAL_PHANTOMS, disp_ind=random.randint(0, len(VAL_SINOS)-1))
            print("="*40)
            save_model_checkpoint(model, optimizer, val_loss, ar, GIT_ROOT/f"data/models/fnobp_ar{ar:.2}_val{val_loss.item()}.pt")
            if val_loss < best_valloss:
                save_model_checkpoint(model, optimizer, val_loss, ar, GIT_ROOT/f"data/models/highscores/{ar:.2}/fnobp_{'-'.join(map(str, hidden_layers))}.pt")
            for i in plt.get_fignums():
                fig = plt.figure(i)
                title = fig._suptitle.get_text() if fig._suptitle is not None else f"fig{i}"
                plt.savefig(f"ar{ar}_fig{i}.png")
            plt.close("all")
        batch_losses = []
        for phantom_batch, sino_batch in dataloader:
            optimizer.zero_grad()

            shift = random.randint(0, geometry.n_projections-1)
            sino_batch = geometry.rotate_sinos(sino_batch, shift)
            la_sinos, known_angles = geometry.zero_cropp_sinos(sino_batch, ar, 0) #maybe start randomizing firts index soon
            fsinos = model.get_extrapolated_filtered_sinos(la_sinos, known_angles)
            fsinos = geometry.rotate_sinos(fsinos, -shift)
            recons = geometry.project_backward(fsinos)
            # recons = model.forward(la_sinos, known_angles, use_relu=relu_in_training)

            loss = MSE(recons, phantom_batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        print("Epoch:", epoch, "mse recon loss:", mean(batch_losses))
