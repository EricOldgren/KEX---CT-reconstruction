import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from utils.polynomials import Legendre, POLYNOMIAL_FAMILY_MAP
from geometries import FBPGeometryBase, DTYPE, CDTYPE, DEVICE, enforce_moment_constraints
from models.modelbase import FBPModelBase


MODEL_DIM = 512


def complex_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    raise NotImplementedError("not decided on activation function yet:/")

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    d = Q.shape[-1]
    activations = F.softmax(Q@K.T / np.sqrt(d), dim=-1)
    return activations@V

class SingleHeadAttention(torch.nn.Module):

    def __init__(self, d_qin: int, d_kin: int, d_vin: int, query_key_dim = 512, value_dim = 512):
        super().__init__()

        self.Wq = nn.Linear(d_qin, query_key_dim, bias=False)
        self.Wk = nn.Linear(d_kin, query_key_dim, bias=False)
        self.Wv = nn.Linear(d_vin, value_dim, bias=False)
        
    def forward(self, qin: torch.Tensor, kin: torch.Tensor, vin: torch.Tensor):
        Q = self.Wq(qin)
        K = self.Wk(kin)
        V = self.Wv(vin)

        return scaled_dot_product_attention(Q, K, V)
    
class MultiheadAttention(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

class AttentionEncoder(FBPModelBase):

    def __init__(self, geometry: FBPGeometryBase, ar: float, M: int, K: int, n_keyvals: int, PolynomialFamilyKey = Legendre.key, n_blocks = 4, strict_moments = True):
        super().__init__()
        self._init_args = (geometry, ar, n_keyvals, n_blocks, strict_moments)
        self.geometry = geometry
        self.ar = ar
        self.M, self.K = M, K
        self.PolynomialFamily = POLYNOMIAL_FAMILY_MAP[PolynomialFamilyKey]
        self.strict_moments = strict_moments

        in_dim = geometry.n_known_projections(ar)*geometry.projection_size
        out_dim = M*K

        self.key_list = nn.ParameterList(
            [nn.Parameter(torch.randn((n_keyvals, in_dim), dtype=DTYPE, device=DEVICE)/(in_dim), requires_grad=True)] +\
            [nn.Parameter(torch.randn((n_keyvals, out_dim), dtype=DTYPE, device=DEVICE)/(out_dim), requires_grad=True) for _ in range(n_blocks-1)]
        )
        self.val_list = nn.ParameterList(
            [nn.Parameter(torch.randn((n_keyvals, out_dim), dtype=DTYPE, device=DEVICE)/(out_dim), requires_grad=True) for _ in range(n_blocks)]
        )

        self.ffn = nn.Sequential(
            # nn.Linear(out_dim, 2048, dtype=DTYPE, device=DEVICE),
            # nn.ReLU(),
            nn.Linear(out_dim, M*K, dtype=DTYPE, device=DEVICE)
        )
        self.ffn_imag = nn.Sequential(
            # nn.Linear(out_dim, 2048, dtype=DTYPE, device=DEVICE),
            # nn.ReLU(),
            nn.Linear(out_dim, M*K, dtype=DTYPE, device=DEVICE)
        )

    def get_init_torch_args(self):
        return self._init_args
    
    def get_extrapolated_sinos(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out = None):
        inp = sinos[:, known_angles]
        N, Nb, Nu = inp.shape
        out = inp.reshape(N, -1)
        for K, V in zip(self.key_list, self.val_list):
            out = scaled_dot_product_attention(out, K, V)# + out
            out = F.layer_norm(out, normalized_shape=out.shape[-1:])

        coefficients = (self.ffn(out) + 1j*self.ffn_imag(out)).reshape(N, self.M, self.K)
        if self.strict_moments:
            enforce_moment_constraints(coefficients)

        return self.geometry.synthesise_series(coefficients, self.PolynomialFamily)
    
    def get_extrapolated_filtered_sinos(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out = None):
        return self.geometry.inverse_fourier_transform(self.geometry.fourier_transform(self.get_extrapolated_sinos(sinos, known_angles, angles_out)*self.geometry.jacobian_det)*self.geometry.ram_lak_filter())
    
    def forward(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out = None):
        return self.geometry.fbp_reconstruct(self.get_extrapolated_sinos(sinos, known_angles, angles_out))



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
    # PHANTOMS = VALIDATION_PHANTOMS = get_htc2022_train_phantoms()
    print("phantoms are loaded")
    SINOS = geometry.project_forward(PHANTOMS)
    print("sinos are calculated")
    dataset = TensorDataset(PHANTOMS, SINOS)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    M, K = 120, 60

    model = AttentionEncoder(geometry, ar, M, K, 300, Legendre.key, 1)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)

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

        print("epoch:", epoch, "sino loss:", mean(sino_losses))

    VALIDATION_SINOS = geometry.project_forward(VALIDATION_PHANTOMS)
    _, known_angles = geometry.zero_cropp_sinos(VALIDATION_SINOS, ar, 0)

    disp_ind = 1
    save_model_checkpoint(model, optimizer, mse_sinos, ar, GIT_ROOT / f"data/models/atentionencoderv1_{mean(sino_losses)}.pt")
    plot_model_progress(model, VALIDATION_SINOS, known_angles, VALIDATION_PHANTOMS, disp_ind=disp_ind)
    
    for i in plt.get_fignums():
        fig = plt.figure(i)
        title = fig._suptitle.get_text() if fig._suptitle is not None else f"fig{i}"
        plt.savefig(f"{title}.png")