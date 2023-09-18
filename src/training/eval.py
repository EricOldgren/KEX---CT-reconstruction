import torch

# matplotlib.use("WebAgg")
import matplotlib.pyplot as plt
from utils.data import get_htc2022_train_phantoms, GIT_ROOT
from utils.polynomials import Legendre

from geometries import FBPGeometryBase, enforce_moment_constraints, naive_sino_filling
from models.modelbase import load_model_checkpoint, plot_model_progress, evaluate_batches
from models.SerieBPs.series_bp1 import Series_BP
from models.SerieBPs.fnoencoder import FNO_Encoder


# checkpoint = load_model_checkpoint(GIT_ROOT / "data/models/fnoencoderv1_4.292833654247985.pt", FNO_Encoder)
model = FNO_Encoder.load("/home/ubuntu/KEX---CT-reconstruction/data/models/fnoencoderv1_8.027480158031496.pt")
ar = model.ar
# model: FNO_Encoder = checkpoint.model
print(model)
geometry = model.geometry

PHANTOMS = get_htc2022_train_phantoms()
SINOS = geometry.project_forward(PHANTOMS)

la_sinos, known_angles = geometry.zero_cropp_sinos(SINOS, ar, 0)

reflected_sinos, known_region = geometry.reflect_fill_sinos(la_sinos+0, known_angles)
reflected_sinos = naive_sino_filling(reflected_sinos, known_region.sum(dim=-1)>0)
proj_coeffs = geometry.series_expand(reflected_sinos, Legendre, 120, 60)
enforce_moment_constraints(proj_coeffs)
proj_exp_sinos = geometry.synthesise_series(proj_coeffs, Legendre)

recons_ptoj = geometry.fbp_reconstruct(proj_exp_sinos)

exp_sinos = model.get_extrapolated_sinos(la_sinos, known_angles)
recons = geometry.fbp_reconstruct(exp_sinos)

disp_ind = 1
sin_fig_proj, sino_mse_proj = evaluate_batches(proj_exp_sinos, SINOS, disp_ind, "sinos proj")
recon_fig_proj, recon_mse_proj = evaluate_batches(recons_ptoj, PHANTOMS, disp_ind, "recons proj")
bin_fig_proj, bin_mse_proj = evaluate_batches((recons_ptoj>0.5).to(torch.float), PHANTOMS, disp_ind, "bin recon proj")

print("sino proj MSE:", sino_mse_proj)
print("recon proj MSE:", recon_mse_proj)
print("bin proj MSE:", bin_mse_proj)

sin_fig_proj.show()
recon_fig_proj.show()
bin_fig_proj.show()


sin_fig, sino_mse = evaluate_batches(exp_sinos.detach(), SINOS, disp_ind, "sinos")
recon_fig, recon_mse = evaluate_batches(recons.detach(), PHANTOMS, disp_ind, "recons")
bin_fig, bin_mse = evaluate_batches((recons.detach()>0.5).to(torch.float), PHANTOMS, disp_ind, "bin recon")

print("sino MSE:", sino_mse)
print("recon MSE:", recon_mse)
print("bin MSE:", bin_mse)

sin_fig.show()
recon_fig.show()
bin_fig.show()

for i in plt.get_fignums():
    fig = plt.figure(i)
    title = fig._suptitle.get_text() if fig._suptitle is not None else f"fig{i}"
    plt.savefig(f"{title}.png")



