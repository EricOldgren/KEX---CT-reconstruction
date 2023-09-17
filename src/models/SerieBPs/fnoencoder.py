import torch
import torch.nn.functional as F

from utils.fno_1d import FNO1d
from utils.polynomials import Legendre, POLYNOMIAL_FAMILY_MAP
from geometries import FBPGeometryBase, DEVICE, DTYPE, CDTYPE, enforce_moment_constraints

from models.modelbase import FBPModelBase


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
        self.fno_pk = FNO1d(channels_phi//2, channels_s, K, layer_widths=hidden_layers_pk, dtype=DTYPE).to(DEVICE)
        self.fno_sm = FNO1d(K//2, channels_phi, M, layer_widths=hidden_layers_sm, dtype=DTYPE).to(DEVICE)
        self.fno_pk_imag = FNO1d(channels_phi//2, channels_s, K, layer_widths=hidden_layers_pk, dtype=DTYPE).to(DEVICE)
        self.fno_sm_imag = FNO1d(K//2, channels_phi, M, layer_widths=hidden_layers_sm, dtype=DTYPE).to(DEVICE)

        self.FFN = torch.nn.Sequential(
            torch.nn.Linear(M*K, 100, device=DEVICE, dtype=DTYPE),
            torch.nn.ReLU(),
            torch.nn.Linear(100, M*K, device=DEVICE, dtype=DTYPE)
        )
        self.FFN_imag = torch.nn.Sequential(
            torch.nn.Linear(M*K, 100, device=DEVICE, dtype=DTYPE),
            torch.nn.ReLU(),
            torch.nn.Linear(100, M*K, device=DEVICE, dtype=DTYPE)
        )

    def get_init_torch_args(self):
        return self._init_args
    
    def get_extrapolated_sinos(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out = None):
        inp = sinos[:, known_angles]
        N, Nb, Nu = inp.shape

        out = self.fno_pk(inp.permute(0,2,1)) # shape: N x K x Nb
        out:torch.Tensor = self.fno_sm(out.permute(0,2,1)) # shape: N x M x K
        out = out + self.FFN(out.reshape(N,-1)).reshape(N, self.M, self.K)
        out_imag = self.fno_pk_imag(inp.permute(0,2,1))
        out_imag:torch.Tensor = self.fno_sm_imag(out_imag.permute(0,2,1))
        out_imag = out_imag + self.FFN_imag(out_imag.reshape(N, -1)).reshape(N, self.M, self.K)

        coefficients = out + 1j*out_imag
        coefficients = coefficients
        if self.strict_moments:
            enforce_moment_constraints(coefficients)

        return self.geometry.synthesise_series(coefficients, self.PolynomialFamily)
    
    def get_extrapolated_filtered_sinos(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out = None):
        return self.geometry.inverse_fourier_transform(self.geometry.fourier_transform(self.get_extrapolated_sinos(sinos, known_angles)*self.geometry.jacobian_det)*self.geometry.ram_lak_filter())
    
    def forward(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out = None):
        return self.geometry.project_backward(self.get_extrapolated_filtered_sinos(sinos, known_angles, angles_out))

    


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

    model = FNO_Encoder(geometry, ar, M, K, hidden_layers_sm=[120, 150, 120], hidden_layers_pk=[100, 100, 100], polynomial_family_key=Legendre.key)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

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
    save_model_checkpoint(model, optimizer, mse_sinos, ar, GIT_ROOT / f"data/models/fnoencoderv1_{mean(sino_losses)}.pt")
    plot_model_progress(model, VALIDATION_SINOS, known_angles, VALIDATION_PHANTOMS, disp_ind=disp_ind)
    
    for i in plt.get_fignums():
        fig = plt.figure(i)
        title = fig._suptitle.get_text() if fig._suptitle is not None else f"fig{i}"
        plt.savefig(f"{title}.png")
    
    
