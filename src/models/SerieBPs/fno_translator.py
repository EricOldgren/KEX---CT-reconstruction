import torch
import torch.nn.functional as F
from math import sqrt

from utils.tools import pacth_split_image_batch, merge_patches
from utils.fno_1d import FNO1d
from utils.polynomials import Legendre, POLYNOMIAL_FAMILY_MAP
from geometries import FBPGeometryBase, DEVICE, DTYPE, CDTYPE, enforce_moment_constraints

from models.modelbase import FBPModelBase

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    d = Q.shape[-1]
    activations = F.softmax(Q@K.T / sqrt(d), dim=-1)
    return activations@V

class MultiHeadAttention(torch.nn.Module):

    def __init__(self, qin_dim: int, kin_dim: int, vin_dim: int, nheads = 8, dk = 512):
        super().__init__()
        assert dk % nheads == 0
        self.nheads = nheads
        self.dk = dk
        self.head_dim = dk // nheads

        self.Wqs = torch.nn.Parameter(torch.randn((nheads, self.head_dim, qin_dim), device=DEVICE, dtype=DTYPE)/(self.head_dim*qin_dim), requires_grad=True)
        self.Wks = torch.nn.Parameter(torch.randn((nheads, self.head_dim, kin_dim), device=DEVICE, dtype=DTYPE)/(self.head_dim*kin_dim), requires_grad=True)
        self.Wvs = torch.nn.Parameter(torch.randn((nheads, self.head_dim, vin_dim), device=DEVICE, dtype=DTYPE)/(self.head_dim*vin_dim), requires_grad=True)

    def forward(self, qin: torch.Tensor, kin: torch.Tensor, vin: torch.Tensor):
        """
            shapes: (...) must coincide
                qin: ... x nqueries x qin_dim
                kin: ... x nvals x kin_dim
                vin: ... x nvals x vin

                return shape: ... x nqueries x dk
        """
        Qs = torch.einsum("...qi,hoi->...hqo", qin, self.Wqs)
        Ks = torch.einsum("...ki,hoi->hko", kin, self.Wks)
        Vs = torch.einsum("...vi,hoi->hvo", vin, self.Wvs)

        return torch.concat([
            scaled_dot_product_attention(Qs[...,hi,:,:], Ks[...,hi,:,:], Vs[...,hi,:,:]) for hi in range(self.nheads)
        ], dim=-1)



class FNO_Translator(FBPModelBase):

    def __init__(self, geometry: FBPGeometryBase, ar: float, M: int, K: int, patch_size: int, hidden_layers_sm = [100,100], hidden_layers_pk = [100,100], polynomial_family_key = Legendre.key, strict_moments = True):
        super().__init__()
        self.geometry = geometry
        self._init_args = (ar, M, K, patch_size, hidden_layers_sm, hidden_layers_pk, polynomial_family_key, strict_moments)
        self.ar = ar
        self.M, self.K = M, K
        self.patch_size = patch_size
        self.strict_moments = strict_moments
        self.PolynomialFamily = POLYNOMIAL_FAMILY_MAP[polynomial_family_key]
        assert M % patch_size == 0
        assert K % patch_size == 0
        self.grid_h, self.grid_w = M // patch_size, K // patch_size
        self.grid_encoding = torch.cartesian_prod(torch.linspace(0,1.0,self.grid_h), torch.linspace(0,1.0,self.grid_w)).to(DEVICE, dtype=DTYPE)


        channels_s = geometry.projection_size
        channels_phi = geometry.n_known_projections(ar)
        self.fno_pk = FNO1d(channels_phi//2, channels_s, K, layer_widths=hidden_layers_pk, dtype=DTYPE).to(DEVICE)
        self.fno_sm = FNO1d(K//2, channels_phi, M, layer_widths=hidden_layers_sm, dtype=DTYPE).to(DEVICE)
        self.multihead = MultiHeadAttention(patch_size**2+2, patch_size**2+2, patch_size**2)
        self.down_rep = torch.nn.Linear(self.multihead.dk, patch_size**2, bias=False, device=DEVICE, dtype=DTYPE)

        self.fno_pk_imag = FNO1d(channels_phi//2, channels_s, K, layer_widths=hidden_layers_pk, dtype=DTYPE).to(DEVICE)
        self.fno_sm_imag = FNO1d(K//2, channels_phi, M, layer_widths=hidden_layers_sm, dtype=DTYPE).to(DEVICE)
        self.multihead_imag = MultiHeadAttention(patch_size**2+2, patch_size**2+2, patch_size**2)
        self.down_rep_imag = torch.nn.Linear(self.multihead.dk, patch_size**2, bias=False, device=DEVICE, dtype=DTYPE)

    def get_init_torch_args(self):
        return self._init_args
    
    def get_extrapolated_sinos(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out = None):
        inp = sinos[:, known_angles]
        N, Nb, Nu = inp.shape

        reflected, _ = self.geometry.reflect_fill_sinos(sinos+0, known_angles)
        patched_coefficients = pacth_split_image_batch(self.geometry.series_expand(reflected, self.PolynomialFamily, self.M, self.K), self.patch_size)

        enc = self.fno_pk(inp.permute(0,2,1)) # shape: N x K x Nb
        enc:torch.Tensor = self.fno_sm(enc.permute(0,2,1)) # shape: N x M x K
        qin = torch.concat([
            F.layer_norm(pacth_split_image_batch(enc, self.patch_size), (self.patch_size**2,)),
            self.grid_encoding[None].repeat(N, 1, 1)
        ], dim=-1) #shape: N x (gh*gw) x patch_size^2 + 2
        kin = torch.concat([patched_coefficients.real, self.grid_encoding[None].repeat(N, 1, 1)], dim=-1)
        out = self.multihead.forward(qin, kin, patched_coefficients.real) # N x (gh*gw) x 512
        out = merge_patches(self.down_rep(out), (self.M, self.K), self.patch_size)
         
        enc_imag = self.fno_pk_imag(inp.permute(0,2,1))
        enc_imag:torch.Tensor = self.fno_sm_imag(enc_imag.permute(0,2,1))
        qin_imag = torch.concat([
            F.layer_norm(pacth_split_image_batch(enc_imag, self.patch_size), (self.patch_size**2,)),
            self.grid_encoding[None].repeat(N, 1, 1)
        ], dim=-1) #shape: N x (gh*gw) x patch_size^2 + 2
        kin_imag = torch.concat([patched_coefficients.imag, self.grid_encoding[None].repeat(N, 1, 1)], dim=-1)
        out_imag = self.multihead_imag.forward(qin_imag, kin_imag, patched_coefficients.imag) # N x (gh*gw) x 512
        out_imag = merge_patches(self.down_rep_imag(out_imag), (self.M, self.K), self.patch_size)

        coefficients = out + out_imag*1j + enc + enc_imag*1j
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

    model = FNO_Translator(geometry, ar, M, K, 12, hidden_layers_sm=[100, 150, 100], hidden_layers_pk=[100, 150, 100], polynomial_family_key=Legendre.key)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
    warmup_steps = 50
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : 30/sqrt(512)*min(1/sqrt(epoch+1), (epoch+1)*warmup_steps**-1.5))

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
    save_model_checkpoint(model, optimizer, mse_sinos, ar, GIT_ROOT / f"data/models/fnotranslatorv1_{mean(sino_losses)}.pt")
    plot_model_progress(model, VALIDATION_SINOS, known_angles, VALIDATION_PHANTOMS, disp_ind=disp_ind)
    
    for i in plt.get_fignums():
        fig = plt.figure(i)
        title = fig._suptitle.get_text() if fig._suptitle is not None else f"fig{i}"
        plt.savefig(f"{title}.png")
    
    
