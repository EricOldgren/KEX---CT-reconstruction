import torch
from torch.utils.data import TensorDataset, DataLoader

# matplotlib.use("WebAgg")
import matplotlib.pyplot as plt
from geometries.data import get_htc2022_train_phantoms, GIT_ROOT, get_htc_traindata, FlatFanBeamGeometry
from utils.polynomials import Legendre, Chebyshev
from utils.tools import MSE, htc_score, segment_imgs

from geometries import FBPGeometryBase, enforce_moment_constraints, naive_sino_filling
from geometries.data import htc_th, htc_mean_attenuation
from models.modelbase import load_model_checkpoint, plot_model_progress, evaluate_batches, save_model_checkpoint
from models.FNOBPs.fnobp import FNO_BP
from models.discriminators.dcnn import load_gan, DCNN
from models.SerieBPs.series_bp1 import Series_BP
from models.SerieBPs.fnoencoder import FNO_Encoder
from models.fbps import AdaptiveFBP
import json
from tqdm import tqdm
import random
from statistics import mean

def path_gen(ar):
    if ar != 161/720:
        return GIT_ROOT / f"data/models/highscores_chain/{ar:.2}/fnobp_80-80-80-80.pt"
    return GIT_ROOT / f"data/models/highscores/{ar:.2}/fnobp_60-60-60.pt"        

# path_gen = lambda ar : GIT_ROOT / f"data/models/highscores_chain/{ar:.2}/fnobp_60-60-60.pt"
ar_lvl_map = {
    181/720: 1, 161/720: 2, 141/720: 3, 121/720: 4, 101/720: 5, 81/720: 6, 61/720: 7
}
SINOS, PHANTOMS = get_htc_traindata()
for ar, lvl in ar_lvl_map.items():
    checkpoint = load_model_checkpoint(path_gen(ar), FNO_BP)
    model: FNO_BP = checkpoint.model
    assert ar == checkpoint.angle_ratio
    geometry: FlatFanBeamGeometry = checkpoint.geometry

    print("Tuning:", ar)
    print("Original validation loss:", checkpoint.loss)
    val2 = plot_model_progress(model, SINOS, ar, PHANTOMS, disp_ind=2)
    print("="*40)
    assert val2 == checkpoint.loss

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

    dataset = TensorDataset(PHANTOMS, SINOS)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    best_valloss = 1.0
    n_epochs = 20

    pbar = tqdm(range(n_epochs), desc="training loop")
    for epoch in pbar:
        batch_losses = []
        batch_recon_mses = []
        for phantom_batch, sino_batch in dataloader:
            optimizer.zero_grad()

            shift = random.randint(0, geometry.n_projections-1)
            sino_batch = geometry.rotate_sinos(sino_batch, shift)
            fsinos_gt = geometry.inverse_fourier_transform(geometry.fourier_transform(sino_batch*geometry.jacobian_det)*geometry.ram_lak_filter()/2)
            la_sinos, known_angles = geometry.zero_cropp_sinos(sino_batch, ar, 0)
            fsinos = model.get_extrapolated_filtered_sinos(la_sinos, known_angles)
            mse_fsinos = MSE(fsinos, fsinos_gt)
            fsinos = geometry.rotate_sinos(fsinos, -shift)
            recons = geometry.project_backward(fsinos)
            mse_recons = MSE(recons, phantom_batch)

            loss = mse_fsinos + mse_recons
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            batch_recon_mses.append(mse_recons.item())
            pbar.set_description("training loss:", mean(batch_losses))

    save_model_checkpoint(model, optimizer, mse_recons, ar, path_gen(ar).parent/"tuned.pt")




    # for i in range(3):
    #     plt.figure()
    #     plt.subplot(121)
    #     plt.imshow((recons[i].cpu()>htc_th).cpu())
    #     plt.subplot(122)
    #     plt.imshow(TEST_PHANTOMS[i].cpu())


    # for i in plt.get_fignums(): #Use this loop on a remote computer
    #     fig = plt.figure(i)
    #     title = fig._suptitle.get_text() if fig._suptitle is not None else f"fig{i}"
    #     plt.savefig(f"{title}.png")



