import torch
import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt

from models.modelbase import plot_model_progress, evaluate_batches
from models.FNOBPs.fnobp import FNO_BP, DTYPE, DEVICE
from models.fbps import FBP, AdaptiveFBP as AFBP


# model_path = "data/models/fnobp_draft1.0.pt"
# model = FNO_BP.load(model_path)
ar = 0.25
afbp = AFBP.load_checkpoint("data/models/afbp_ar0.25.pt").model
fno_bp = FNO_BP.load_checkpoint("data/models/fno_bp_ar0.25.pt").model
geometry = afbp.geometry
fbp = FBP(geometry)


models = [fbp, afbp, fno_bp]

PHANTOMS = torch.stack(torch.load("data/HTC2022/HTCTrainingPhantoms.pt")).to(DEVICE, dtype=DTYPE)
SINOS = geometry.project_forward(PHANTOMS)


cropped_sinos, known_beta_bools = geometry.zero_cropp_sinos(SINOS, ar, 0)
cropped_sinos = geometry.reflect_fill_sinos(cropped_sinos, known_beta_bools)

disp_ind = 0
for model in models:
    plot_model_progress(model, cropped_sinos, SINOS, PHANTOMS, disp_ind=disp_ind)

plt.show()

