import torch

# matplotlib.use("WebAgg")
import matplotlib.pyplot as plt
from geometries.data import get_htc2022_train_phantoms, GIT_ROOT, get_htc_traindata, get_htc_testdata, HTC2022_GEOMETRY
from utils.polynomials import Legendre, Chebyshev
from utils.tools import MSE, htc_score, segment_imgs

from geometries import FBPGeometryBase, enforce_moment_constraints, naive_sino_filling
from geometries.data import htc_th, htc_mean_attenuation
from models.modelbase import load_model_checkpoint, plot_model_progress, evaluate_batches
from models.FNOBPs.fnobp import FNO_BP
from models.discriminators.dcnn import load_gan, DCNN
from models.SerieBPs.series_bp1 import Series_BP
from models.SerieBPs.fnoencoder import FNO_Encoder
from models.fbps import AdaptiveFBP


ar_lvl_map = {
    181/720: 1, 161/720: 2, 141/720: 3, 121/720: 4, 101/720: 5, 81/720: 6, 61/720: 7
}
checkpoint = load_model_checkpoint(GIT_ROOT/"data/models/fnobp_60-60-60.pt", FNO_BP)
model: FNO_BP = checkpoint.model
ar = checkpoint.angle_ratio
lvl = ar_lvl_map[ar]
geometry = checkpoint.geometry

SINOS, PHANTOMS = get_htc_traindata()

print("Model init args:")
print(model.get_init_torch_args())
print("Original validation loss:", checkpoint.loss)
print("Confirming validation loss (on the same validation set):")
plot_model_progress(model, SINOS, ar, PHANTOMS, disp_ind=2)
print("="*40)

TEST_SINOS, known_angles, shifts, TEST_PHANTOMS = get_htc_testdata(lvl)

recons = []
with torch.no_grad():
    for i in range(3):
        sino = TEST_SINOS[i]
        fsino = model.get_extrapolated_filtered_sinos(sino[None], known_angles)[0]
        recon = torch.nn.functional.relu(HTC2022_GEOMETRY.project_backward(HTC2022_GEOMETRY.rotate_sinos(fsino[None], shifts[i])))[0]
        recons.append(recon)

recons = torch.stack(recons)
print("Test MSE:", MSE(recons, TEST_PHANTOMS*htc_mean_attenuation))
print("HTC score naive_thresh:", htc_score(recons>htc_th, TEST_PHANTOMS), "in total:", htc_score(recons>htc_th, TEST_PHANTOMS).sum())
otsu_scores = htc_score(segment_imgs(recons), TEST_PHANTOMS)
print("HTC score otsu_thresh:", otsu_scores,"in total:", otsu_scores.sum())

for i in range(3):
    plt.figure()
    plt.subplot(121)
    plt.imshow((recons[i].cpu()>htc_th).cpu())
    plt.subplot(122)
    plt.imshow(TEST_PHANTOMS[i].cpu())

plt.show()
# for i in plt.get_fignums(): #Use this loop on a remote computer
#     fig = plt.figure(i)
#     title = fig._suptitle.get_text() if fig._suptitle is not None else f"fig{i}"
#     plt.savefig(f"{title}.png")



