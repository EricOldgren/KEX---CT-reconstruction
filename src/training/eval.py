import torch

# matplotlib.use("WebAgg")
import matplotlib.pyplot as plt
from utils.data import get_htc2022_train_phantoms, GIT_ROOT

from models.modelbase import load_model_checkpoint, plot_model_progress, evaluate_batches
from models.SerieBPs.series_bp1 import Series_BP

checkpoint = load_model_checkpoint(GIT_ROOT / "data/models/serries_bpv1.1_sino_mse_96.10559844970703.pt", Series_BP)
model: Series_BP = checkpoint.model
geometry = model.geometry
ar = checkpoint.angle_ratio

PHANTOMS = get_htc2022_train_phantoms()
SINOS = geometry.project_forward(PHANTOMS)

la_sinos, known_angles = geometry.zero_cropp_sinos(SINOS, ar, 0)
exp_sinos = model.get_extrapolated_sinos(la_sinos, known_angles)
recons = geometry.fbp_reconstruct(exp_sinos)

disp_ind = 1
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



