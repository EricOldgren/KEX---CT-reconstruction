import torch
import scipy
import matplotlib.pyplot as plt
import json

from utils.tools import GIT_ROOT, DEVICE, MSE, htc_score, segment_imgs

from geometries import HTC2022_GEOMETRY
from geometries.data import get_htc_traindata, htc_th

from models.modelbase import load_model_checkpoint, plot_model_progress
from models.FNOBPs.fnobp import FNO_BP

sino_paths = (
    "data/HTC2022/TestData/htc2022_01a_full.mat",
    "data/HTC2022/TestData/htc2022_01b_full.mat",
    "data/HTC2022/TestData/htc2022_01c_full.mat"
)
full_sinos = []
for p in sino_paths:
    data = scipy.io.loadmat(GIT_ROOT/p)["CtDataFull"][0,0]
    full_sino = torch.tensor(data["sinogram"])[:720].to(DEVICE).to(torch.float32)
    assert full_sino.shape == (720, HTC2022_GEOMETRY.projection_size)
    full_sinos.append(full_sino)

full_sinos = torch.stack(full_sinos)
gt_phantoms = HTC2022_GEOMETRY.fbp_reconstruct(full_sinos)
gt_seg_phantoms = segment_imgs(gt_phantoms)

def path_gen(ar):
    # if ar != 161/720:
        # return GIT_ROOT / f"data/models/highscores/{ar:.2}/afbp.pt"
    return GIT_ROOT / f"data/highscores/{ar:.2}/fnobp_60-60-60.pt"      
save_gen = lambda ar : GIT_ROOT/f"data/htc_results_samephantom/{ar:.2}"
ar_lvl_map = {
    181/720: 1, 161/720: 2, 141/720: 3, 121/720: 4, 101/720: 5, 81/720: 6, 61/720: 7
}
SINOS, PHANTOMS = get_htc_traindata()
scores, scores_using_otsu = [], []
otsu_scores_mom = []

for ar, lvl in ar_lvl_map.items():
    checkpoint = load_model_checkpoint(path_gen(ar), FNO_BP)
    model: FNO_BP = checkpoint.model
    model = model.to(DEVICE)
    assert ar == checkpoint.angle_ratio
    geometry = checkpoint.geometry

    print("Original validation loss:", checkpoint.loss)
    val2 = plot_model_progress(model, SINOS, ar, PHANTOMS, disp_ind=2)
    print("="*40)
    # assert val2 == checkpoint.loss, f"original loss:{ checkpoint.loss}, current: {val2}"

    with torch.no_grad():
        la_sinos, known_angles = geometry.zero_cropp_sinos(full_sinos, ar, 0)
        fsinos = model.get_extrapolated_filtered_sinos(la_sinos, known_angles)
        model_recons = torch.relu(HTC2022_GEOMETRY.project_backward(fsinos))
        la_fbp_recons = geometry.fbp_reconstruct(la_sinos)
        la_mom_recons = geometry.fbp_reconstruct(model.get_extrapolated_sinos(la_sinos, known_angles))

    print("Test MSE:", MSE(model_recons, gt_phantoms))
    print("HTC score naive_thresh:", htc_score(model_recons>htc_th, gt_seg_phantoms), "in total:", htc_score(model_recons>htc_th, gt_seg_phantoms).sum())
    otsu_scores = htc_score(segment_imgs(model_recons), gt_seg_phantoms)
    otsu_mom_scores = htc_score(segment_imgs(la_mom_recons), gt_seg_phantoms)
    print("HTC score otsu_thresh:", otsu_scores,"in total:", otsu_scores.sum())

    scores.append(htc_score(model_recons>htc_th, gt_phantoms>htc_th).sum().item())
    scores_using_otsu.append(otsu_scores.sum().item())
    otsu_scores_mom.append(otsu_mom_scores.sum().item())
    save_path = save_gen(ar)
    if not save_path.exists(): save_path.mkdir(parents=True)

    torch.save(model_recons, save_path/"pred_model.pt")
    torch.save(la_fbp_recons, save_path/"la_fbp.pt")
    torch.save(la_mom_recons, save_path/"la_mom_fbb.pt")

    plt.clf()
    plt.subplot(221)
    plt.imshow(gt_phantoms[0].cpu())
    plt.subplot(222)
    plt.imshow(model_recons[0].cpu())
    plt.subplot(223)
    plt.imshow(la_mom_recons[0].cpu())
    plt.subplot(224)
    plt.imshow(la_fbp_recons[0].cpu())

    plt.savefig(f"{lvl=}.png")

torch.save(gt_phantoms, save_gen(0.1).parent/"gt.pt")

(save_gen(10.0).parent / "score.json").write_text(json.dumps({
    "scores": scores,
    "scores_using_otsu:": scores_using_otsu,
    "mom_fbp_otsu_score": otsu_scores_mom
}))