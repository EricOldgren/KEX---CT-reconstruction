import torch
from torch.utils.data import TensorDataset, DataLoader
from utils.tools import MSE, GIT_ROOT, DEVICE
from utils.polynomials import Chebyshev
from geometries import HTC2022_GEOMETRY, FBPGeometryBase
from geometries.data import get_synthetic_htc_phantoms, get_htc_traindata
from models.modelbase import FBPModelBase, plot_model_progress, save_model_checkpoint
from models.FNOBPs.fnobp2d import FNOBP2D
import sys
import random
import matplotlib.pyplot as plt
from statistics import mean
from tqdm import tqdm

HIGH_SCORES_FOLDER = GIT_ROOT/"data/models/highscores_fnobp2D"
pt_baseename = "fnobp2D"

PHANTOMS = get_synthetic_htc_phantoms(use_kits=False)
SINOS = HTC2022_GEOMETRY.project_forward(PHANTOMS)
VAL_SINOS, VAL_PHANTOMS = get_htc_traindata()

def train_fno(geometry: FBPGeometryBase, ar: float, M: int = 50, K: int = 50, n_epochs: int = 40):
    print("Training for ar:", ar)

    ridge_reg = 0.01

    model = FNOBP2D(geometry, ar, [40,40,40], modes_y=30, modes_x=30, M=M,K=K, use_base_filter=True).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    dataset = TensorDataset(PHANTOMS, SINOS)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    best_valloss = 1.0

    for epoch in range(n_epochs):
        if ((epoch)%5) == 0:
            print("Validation results:")
            val_loss = plot_model_progress(model, VAL_SINOS, ar, VAL_PHANTOMS, disp_ind=random.randint(0, len(VAL_SINOS)-1))
            print("="*40)
            save_model_checkpoint(model, optimizer, val_loss, ar, GIT_ROOT/f"data/models/{pt_baseename}_ar{ar:.2}_val{val_loss.item()}.pt")
            if val_loss < best_valloss:
                save_model_checkpoint(model, optimizer, val_loss, ar, HIGH_SCORES_FOLDER/f"{ar:.2}/{pt_baseename}.pt")
            for i in plt.get_fignums():
                fig = plt.figure(i)
                title = fig._suptitle.get_text() if fig._suptitle is not None else f"fig{i}"
                plt.savefig(f"ar{ar}_fig{i}.png")
            plt.close("all")
        batch_losses = []
        pbar = tqdm(dataloader, desc="training")
        for phantom_batch, sino_batch in pbar:
            optimizer.zero_grad()

            shift = random.randint(0, geometry.n_projections-1)
            sino_batch = geometry.rotate_sinos(sino_batch, shift)
            la_sinos, known_angles = geometry.zero_cropp_sinos(sino_batch, ar, 0)
            fsinos = model.get_extrapolated_filtered_sinos(la_sinos, known_angles)
            fsinos = geometry.rotate_sinos(fsinos, -shift)
            recons = torch.nn.functional.relu(geometry.project_backward(fsinos))
            loss = MSE(recons, phantom_batch)

            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            pbar.set_description(f"training loss: {mean(batch_losses)}")

        print("Epoch:", epoch, "training loss:", mean(batch_losses))


n_epochs = int(sys.argv[1])

nprojs = [61, 81, 101]
for n in nprojs:
    train_fno(HTC2022_GEOMETRY, n/720, n_epochs=n_epochs)