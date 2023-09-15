import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
 
from utils.polynomials import Legendre, POLYNOMIAL_FAMILY_MAP
from geometries import FBPGeometryBase, DEVICE, DTYPE, CDTYPE, get_moment_mask
from models.modelbase import FBPModelBase, load_model_checkpoint, PathType 


class Series_BP(FBPModelBase):

    def __init__(self, geometry: FBPGeometryBase, ar: float, M: int, K: int, polynomial_family_key: int = Legendre.key, strict_moments = True):
        assert 0 < ar <= 1.0, f"angle ratio, {ar} is invalid"
        super().__init__()
        self.geometry = geometry
        self._init_args = (ar, M, K, polynomial_family_key, strict_moments)

        n_known_angles = geometry.n_known_projections(ar)
        self.M, self.K = M, K
        self.PolynomialFamily = POLYNOMIAL_FAMILY_MAP[polynomial_family_key]
        self.strict_moments = strict_moments

        h, w, c = n_known_angles, geometry.projection_size, 1
        next_c = lambda c : 8 if c == 1 else min(64, c*2)
        conv_layers = []
        while min(h, w) >= 4:
            conv_layers.append(nn.Conv2d(c, next_c(c), (4,4), 2, padding=0, device=DEVICE))
            # conv_layers.append(nn.LeakyReLU(0.2))
            c = next_c(c)
            h = (h-4)//2 + 1
            w = (w-4)//2 + 1
 
        self.moment_mask = get_moment_mask(torch.zeros((1,M,K), device=DEVICE))
        n_coeffs = M*K  #self.moment_mask.count_nonzero()
        self.conv_layers = nn.ModuleList(conv_layers)
        self.lin_out = nn.Linear(64, n_coeffs, dtype=CDTYPE, device=DEVICE)
    
    def get_init_torch_args(self):
        return self._init_args

    def get_extrapolated_sinos(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out: torch.Tensor = None):
        out = sinos[:,None, known_angles]
        N, h, w = sinos.shape
        for i, conv in enumerate(self.conv_layers):
            out = conv(out)
            out = F.leaky_relu(out, 0.2)

        out = torch.mean(out, dim=(-1,-2)) + 0*1j

        coefficients: torch.Tensor = self.lin_out(out).reshape(N, self.M, self.K)
        if self.strict_moments:
            coefficients[:, ~self.moment_mask] *= 0

        return self.geometry.synthesise_series(coefficients, self.PolynomialFamily)
    
    def get_extrapolated_filtered_sinos(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out: torch.Tensor = None):
        sinos = self.get_extrapolated_sinos(sinos, known_angles, angles_out)
        return self.geometry.inverse_fourier_transform(self.geometry.fourier_transform(sinos*self.geometry.jacobian_det)*self.geometry.ram_lak_filter()/2)
    
    def forward(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out: torch.Tensor = None):
        return self.geometry.project_backward(self.get_extrapolated_filtered_sinos(sinos, known_angles, angles_out))
    
    @staticmethod
    def load(path: PathType):
        return load_model_checkpoint(path, Series_BP).model
    

        
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

    model = Series_BP(geometry, ar, M, K, Legendre.key, strict_moments=False)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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
    save_model_checkpoint(model, optimizer, mse_sinos, ar, GIT_ROOT / f"data/models/serries_bp_not_strict_v1.1_sino_mse_{mean(sino_losses)}.pt")
    plot_model_progress(model, VALIDATION_SINOS, known_angles, VALIDATION_PHANTOMS, disp_ind=disp_ind, model_name="SeriesBP_not_strict")
    
    for i in plt.get_fignums():
        fig = plt.figure(i)
        title = fig._suptitle.get_text() if fig._suptitle is not None else f"fig{i}"
        plt.savefig(f"{title}.png")
    




