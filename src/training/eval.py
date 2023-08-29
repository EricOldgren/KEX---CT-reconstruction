import torch
import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt

from utils.data import get_htc2022_train_phantoms, get_kits_train_phantoms
from geometries.geometry_base import mark_cyclic
from models.modelbase import plot_model_progress, evaluate_batches
from models.FNOBPs.fnobp import FNO_BP, DTYPE, DEVICE
from models.fbps import FBP, AdaptiveFBP as AFBP


# model_path = "data/models/fnobp_draft1.0.pt"
# model = FNO_BP.load(model_path)
ar = 0.25
afbp = AFBP.load_checkpoint("data/models/afbp_ar0.25.pt").model
fno_bp = FNO_BP.load_checkpoint("data/models/fno_bp_fanbeamkits_ar0.25.pt").model
geometry = fno_bp.geometry
fbp = FBP(geometry)


models = [fbp, fno_bp]

PHANTOMS = get_kits_train_phantoms()[:10]
SINOS = geometry.project_forward(PHANTOMS)

N_known_angles = int(geometry.n_projections*0.25)
N_angles_out = int(geometry.n_projections*0.5)

start_ind = 0
known_angles = torch.zeros(geometry.n_projections, device=DEVICE, dtype=bool)
out_angles = known_angles.clone()
mark_cyclic(known_angles, start_ind, (start_ind+N_known_angles)%geometry.n_projections)#known_beta_bool is True at angles where sinogram is meassured and false otherwise
mark_cyclic(out_angles, start_ind, (start_ind+N_angles_out)%geometry.n_projections)

disp_ind = 0
for model in models:
    plot_model_progress(model, SINOS, known_angles, out_angles, PHANTOMS, disp_ind=disp_ind)

plt.show()

