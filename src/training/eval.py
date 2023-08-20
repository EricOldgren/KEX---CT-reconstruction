import torch
import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt

from models.modelbase import plot_model_progress, evaluate_batches
from models.FNOBPs.fnobp import FNO_BP, DTYPE, DEVICE
from models.fbps import FBP, AdaptiiveFBP as AFBP

# model_path = "data/models/fnobp_draft1.0.pt"
# model = FNO_BP.load(model_path)
model_path = "data/models/afbp_draft1.0.pt"
model = AFBP.load(model_path)
geometry = model.geometry

PHANTOMS = torch.stack(torch.load("data/HTC2022/HTCTrainingPhantoms.pt")).to(DEVICE, dtype=DTYPE)
SINOS = geometry.project_forward(PHANTOMS)


cropped_sinos, known_beta_bools = geometry.zero_cropp_sinos(SINOS, 0.5, 0)
cropped_sinos = geometry.reflect_fill_sinos(cropped_sinos, known_beta_bools)

fbp = FBP(geometry)
disp_ind = 0
print("="*40)
print("FBP")
plot_model_progress(fbp, cropped_sinos, SINOS, PHANTOMS, disp_ind)
print()
print("="*40)
print("Model")
plot_model_progress(model, cropped_sinos, SINOS, PHANTOMS, disp_ind)

plt.show()

