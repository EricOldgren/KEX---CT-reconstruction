import torch

# matplotlib.use("WebAgg")
import matplotlib.pyplot as plt
from geometries.data import get_htc2022_train_phantoms, GIT_ROOT, get_htc_traindata, get_htc_testdata, HTC2022_GEOMETRY
from utils.polynomials import Legendre, Chebyshev
from utils.tools import MSE, htc_score, segment_imgs, DEVICE

from geometries import FBPGeometryBase, enforce_moment_constraints, naive_sino_filling
from geometries.data import htc_th, htc_mean_attenuation
from models.modelbase import load_model_checkpoint, plot_model_progress, evaluate_batches
from models.FNOBPs.fnobp2 import FNOBP2
from models.fbps import AdaptiveFBP

import json


def path_gen(ar):
    # if ar != 161/720:
        # return GIT_ROOT / f"data/models/highscores/{ar:.2}/afbp.pt"
    return GIT_ROOT / f"data/models/highscores_fno2/{ar:.2}/fnobp2.pt"        
save_gen = lambda ar : GIT_ROOT/f"data/htc_results_fno2/{ar:.2}"
ar_lvl_map = {
    181/720: 1, 161/720: 2, 141/720: 3, 121/720: 4, 101/720: 5, 81/720: 6, 61/720: 7
}
SINOS, PHANTOMS = get_htc_traindata()
scores, scores_using_otsu = [], []
for ar, lvl in ar_lvl_map.items():
    checkpoint = load_model_checkpoint(path_gen(ar), FNOBP2)
    model: FNOBP2 = checkpoint.model
    model = model.to(DEVICE)
    assert ar == checkpoint.angle_ratio
    geometry = checkpoint.geometry

    print("Original validation loss:", checkpoint.loss)
    val2 = plot_model_progress(model, SINOS, ar, PHANTOMS, disp_ind=2)
    print("="*40)
    assert val2 == checkpoint.loss, f"original loss:{ checkpoint.loss}, current: {val2}"

    TEST_SINOS, known_angles, shifts, TEST_PHANTOMS = get_htc_testdata(lvl)

    recons = []
    with torch.no_grad():
        for i in range(3):
            sino = TEST_SINOS[i]
            fsino = model.get_extrapolated_filtered_sinos(sino[None], known_angles)[0]
            recon = torch.sigmoid(HTC2022_GEOMETRY.project_backward(HTC2022_GEOMETRY.rotate_sinos(fsino[None], shifts[i])))[0]
            recons.append(recon)

    recons = torch.stack(recons)
    print("Test MSE:", MSE(recons, TEST_PHANTOMS*htc_mean_attenuation))
    print("HTC score naive_thresh:", htc_score(recons>htc_th, TEST_PHANTOMS), "in total:", htc_score(recons>htc_th, TEST_PHANTOMS).sum())
    otsu_scores = htc_score(segment_imgs(recons), TEST_PHANTOMS)
    print("HTC score otsu_thresh:", otsu_scores,"in total:", otsu_scores.sum())

    scores.append(htc_score(recons>htc_th, TEST_PHANTOMS).sum().item())
    scores_using_otsu.append(otsu_scores.sum().item())
    save_path = save_gen(ar)
    if not save_path.exists(): save_path.mkdir(parents=True)

    torch.save(recons, save_path/"pred.pt")
    torch.save(TEST_PHANTOMS, save_path/"gt.pt")

(save_gen(10.0).parent / "score.json").write_text(json.dumps({
    "scores": scores,
    "scores_using_otsu:": scores_using_otsu
}))

