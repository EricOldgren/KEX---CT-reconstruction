import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
 
from utils.polynomials import Legendre, POLYNOMIAL_FAMILY_MAP
from geometries import FBPGeometryBase, DEVICE, DTYPE, CDTYPE, get_moment_mask, enforce_moment_constraints, naive_sino_filling
from models.modelbase import FBPModelBase, load_model_checkpoint, PathType 


class UNet(torch.nn.Module):

    def __init__(self, h: int, w: int, cmin = 8, cmax = 64) -> None:
        super().__init__()
        h, w, c = M, K, 1
        next_c = lambda c : cmin if c == 1 else min(cmax, c*2)
        conv_layers = []
        deconv_layers = []
        while min(h, w) >= 4:
            conv_layers.append(nn.Conv2d(c, next_c(c), (4,4), 2, padding=1, device=DEVICE))
            deconv_layers.append(nn.ConvTranspose2d(next_c(c)*2, c, (4,4), 2, padding=1, device=DEVICE))
            c = next_c(c)
            h = h // 2
            w = w // 2
        conv_layers.append(nn.Conv2d(c, c, (h, w), padding=0, device=DEVICE))
        deconv_layers.append(nn.ConvTranspose2d(c, c, (h, w), padding=0, device=DEVICE))

        self.conv_layers = nn.ModuleList(conv_layers)
        self.deconv_layers = nn.ModuleList(deconv_layers[::-1])

    def forward(self, inp: torch.Tensor):

        N, h, w = inp.shape
        out = inp[:, None]
        encs = []
        for i, conv in enumerate(self.conv_layers):
            out = conv(out)
            out = F.leaky_relu(out, 0.2)
            encs.append(out)
        encs.reverse()
        for i, (enc, deconv) in enumerate(zip(encs, self.deconv_layers)):
            if i > 0:
                out = torch.concat([enc, out], dim=-3)
            out = deconv(out)
            out = F.leaky_relu(out, 0.2)

        assert out.shape[1] == 1 #one channel
        return out[:, 0]


class UNetBP(FBPModelBase):

    def __init__(self, geometry: FBPGeometryBase, ar: float, M: int, K: int, polynomial_family_key: int = Legendre.key, strict_moments = True):
        assert 0 < ar <= 1.0, f"angle ratio, {ar} is invalid"
        super().__init__()
        self.geometry = geometry
        self._init_args = (ar, M, K, polynomial_family_key, strict_moments)

        self.M, self.K = M, K
        self.PolynomialFamily = POLYNOMIAL_FAMILY_MAP[polynomial_family_key]
        self.strict_moments = strict_moments

        self.unet_real = UNet(M, K)
        self.unet_imag = UNet(M, K)
    
    def get_init_torch_args(self):
        return self._init_args

    def get_extrapolated_sinos(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out: torch.Tensor = None):
         
        reflected_sinos, known_region = geometry.reflect_fill_sinos(la_sinos+0, known_angles)
        reflected_sinos = naive_sino_filling(reflected_sinos, known_region.sum(dim=-1)>0)
        proj_coeffs = geometry.series_expand(reflected_sinos, Legendre, self.M, self.K)
        enforce_moment_constraints(proj_coeffs)

        out_real = self.unet_real(proj_coeffs.real)
        out_imag = self.unet_imag(proj_coeffs.imag)
        
        coefficients = out_real + 1j*out_imag
        if self.strict_moments:
            enforce_moment_constraints(coefficients)

        return self.geometry.synthesise_series(coefficients, self.PolynomialFamily)
    
    def get_extrapolated_filtered_sinos(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out: torch.Tensor = None):
        sinos = self.get_extrapolated_sinos(sinos, known_angles, angles_out)
        return self.geometry.inverse_fourier_transform(self.geometry.fourier_transform(sinos*self.geometry.jacobian_det)*self.geometry.ram_lak_filter()/2)
    
    def forward(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out: torch.Tensor = None):
        return self.geometry.project_backward(self.get_extrapolated_filtered_sinos(sinos, known_angles, angles_out))
    
    @staticmethod
    def load(path: PathType):
        return load_model_checkpoint(path, UNetBP).model
    

        
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

    M, K = 128, 64

    model = UNetBP(geometry, ar, M, K, Legendre.key, strict_moments=True)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
    warmup_steps = 50
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : min((epoch+1)**-0.5, (epoch+1)*warmup_steps**-1.5))

    n_epochs = 200
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
    save_model_checkpoint(model, optimizer, mse_sinos, ar, GIT_ROOT / f"data/models/unetbp_sino_mse_{mean(sino_losses)}.pt")
    plot_model_progress(model, VALIDATION_SINOS, known_angles, VALIDATION_PHANTOMS, disp_ind=disp_ind)
    
    for i in plt.get_fignums():
        fig = plt.figure(i)
        title = fig._suptitle.get_text() if fig._suptitle is not None else f"fig{i}"
        plt.savefig(f"{title}.png")
    




