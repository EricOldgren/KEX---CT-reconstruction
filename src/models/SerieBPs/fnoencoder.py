import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tools import PathType
from utils.fno_1d import FNO1d
from utils.layers import MultiHeadAttention, positional_encoding
from utils.polynomials import Legendre, POLYNOMIAL_FAMILY_MAP
from geometries import FBPGeometryBase, DEVICE, DTYPE, CDTYPE, enforce_moment_constraints

from models.modelbase import FBPModelBase, load_model_checkpoint

class _EncoderLayer(nn.Module):

    def __init__(self, M: int, K: int, channels_s: int, channels_phi: int, hidden_layers_sm, hidden_layers_pk):
        super().__init__()

        self.fno_pk = FNO1d(channels_phi//2, channels_s, K, layer_widths=hidden_layers_pk, dtype=DTYPE).to(DEVICE)
        self.fno_sm = FNO1d(K//2, channels_phi, M, layer_widths=hidden_layers_sm, dtype=DTYPE).to(DEVICE)

        self.positional_mask = positional_encoding(M, K)
        self.attention_layer = MultiHeadAttention(K, K, K, dout=K)

    def forward(self, inp: torch.Tensor):
        N, Nu, Nb = inp.shape

        out = self.fno_pk(inp.permute(0,2,1)) # shape: N x K x Nb
        out:torch.Tensor = self.fno_sm(out.permute(0,2,1)) # shape: N x M x K
        pos_masked  = out + self.positional_mask
        assert not pos_masked.isnan().any()
        return out + self.attention_layer.forward(pos_masked, pos_masked, pos_masked)

class FNO_Encoder(FBPModelBase):

    def __init__(self, geometry: FBPGeometryBase, ar: float, M: int, K: int, hidden_layers_sm = [40,40], hidden_layers_pk = [40,40], polynomial_family_key = Legendre.key, strict_moments = True):
        super().__init__()
        self.geometry = geometry
        self._init_args = (ar, M, K, hidden_layers_sm, hidden_layers_pk, polynomial_family_key, strict_moments)
        self.ar = ar
        self.M, self.K = M, K
        self.strict_moments = strict_moments
        self.PolynomialFamily = POLYNOMIAL_FAMILY_MAP[polynomial_family_key]

        channels_s = geometry.projection_size
        channels_phi = geometry.n_known_projections(ar)
        self.encoder_real = _EncoderLayer(M, K, channels_s, channels_phi, hidden_layers_sm, hidden_layers_pk)
        self.encoder_imag = _EncoderLayer(M, K, channels_s, channels_phi, hidden_layers_sm, hidden_layers_pk)

    def get_init_torch_args(self):
        return self._init_args
    
    def get_coefficients(self, sinos: torch.Tensor, known_angles: torch.Tensor):
        assert not sinos.isnan().any()
        inp = sinos[:, known_angles]
        N, Nb, Nu = inp.shape

        out = self.encoder_real.forward(inp)
        out_imag = self.encoder_imag.forward(inp)

        coefficients = out + 1j*out_imag
        assert not coefficients.isnan().any()
        if self.strict_moments:
            enforce_moment_constraints(coefficients)

        return coefficients

    def get_extrapolated_sinos(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out = None):
        
        coefficients = self.get_coefficients(sinos, known_angles)
        return self.geometry.synthesise_series(coefficients, self.PolynomialFamily)
    
    def get_extrapolated_filtered_sinos(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out = None):
        return self.geometry.inverse_fourier_transform(self.geometry.fourier_transform(self.get_extrapolated_sinos(sinos, known_angles)*self.geometry.jacobian_det)*self.geometry.ram_lak_filter())
    
    def forward(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out = None):
        return self.geometry.project_backward(self.get_extrapolated_filtered_sinos(sinos, known_angles, angles_out))

    @staticmethod
    def load(path: PathType)->"FNO_Encoder":
        return load_model_checkpoint(path, FNO_Encoder).model
    


if __name__ == "__main__":
    from torch.utils.data import TensorDataset, DataLoader
    from utils.tools import MSE, htc_score
    from utils.polynomials import Legendre, Chebyshev
    from utils.data import get_htc2022_train_phantoms, get_htc_trainval_phantoms, GIT_ROOT
    from geometries import HTC2022_GEOMETRY
    from models.modelbase import plot_model_progress, save_model_checkpoint
    import matplotlib
    import matplotlib.pyplot as plt
    from statistics import mean
    ar = 0.25
    geometry = HTC2022_GEOMETRY
    PHANTOMS, VALIDATION_PHANTOMS = get_htc_trainval_phantoms()
    print("phantoms are loaded")
    SINOS = geometry.project_forward(PHANTOMS)
    print("sinos are calculated")
    dataset = TensorDataset(PHANTOMS, SINOS)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    M, K = 64, 64

    model = FNO_Encoder(geometry, ar, M, K, hidden_layers_sm=[100, 100, 100], hidden_layers_pk=[100, 100, 100], polynomial_family_key=Legendre.key, strict_moments=False)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
    warmup_steps = 50
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : 30*512**-0.5*min((epoch+1)**-0.5, (epoch+1)*warmup_steps**-1.5))

    n_epochs = 150
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
    save_model_checkpoint(model, optimizer, mse_sinos, ar, GIT_ROOT / f"data/models/fnoencoderv1_{mean(sino_losses)}.pt")
    plot_model_progress(model, VALIDATION_SINOS, known_angles, VALIDATION_PHANTOMS, disp_ind=disp_ind)
    
    for i in plt.get_fignums():
        fig = plt.figure(i)
        title = fig._suptitle.get_text() if fig._suptitle is not None else f"fig{i}"
        plt.savefig(f"{title}.png")
    
    
