import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from statistics import mean

from utils.data import get_htc_trainval_phantoms
from utils.tools import MSE
from utils.polynomials import Legendre
from geometries import HTC2022_GEOMETRY, DEVICE, DTYPE

from models.SerieBPs.fnoencoder import FNO_Encoder
from models.modelbase import load_model_checkpoint, save_model_checkpoint
from models.discriminators.dcnn import DCNN, concat_pred_inp

geometry = HTC2022_GEOMETRY
PHANTOMS, VALIDATION_PHANTOMS = get_htc_trainval_phantoms()
SINOS = geometry.project_forward(PHANTOMS)
ar = 0.25
M, K = 64, 64

G = FNO_Encoder(geometry, ar, M, K, [100,100], [100,100], Legendre.key)
# G = FNO_Encoder.load("/home/ubuntu/KEX---CT-reconstruction/data/models/fnoencoderv1_6.143538619385285.pt")
assert G.strict_moments
D = DCNN(geometry.n_known_projections(ar), geometry.projection_size, 2)

print("Generatot:", G)
print("Disctriminator:", D)

dataset = TensorDataset(PHANTOMS, SINOS)
dataloeder = DataLoader(dataset, batch_size=8)

l1, l2 = 0.03, 1.0
eps = 1e-5

def epoch_f(it):
    return int(it**0.5) + 2
n_iters = 300
for it in range(n_iters):
    minimizer = it % 2
    print("Iteration", it, "Training ", "Generator" if minimizer else "Discriminator")
    if minimizer:
        start_iters = 10
        lr = 3e-4 * min((it+1)*start_iters**-1.5, (it+1)**-0.5)
        optimizer = torch.optim.Adam(G.parameters(), lr=lr)
    else:
        start_iters = 10
        lr = 1e-3 * min((it+1)*start_iters**-1.5, (it+1)**-0.5)
        optimizer = torch.optim.Adam(D.parameters(), lr=lr, maximize=True)

    for epoch in range(epoch_f(it)):

        Lgd, Ls, Lr = [], [], []
        for phantom_batch, sino_batch in dataloeder:
            optimizer.zero_grad()

            la_sinos, known_angles = geometry.zero_cropp_sinos(sino_batch, ar, 0)

            exp_sinos = G.get_extrapolated_sinos(la_sinos, known_angles)
            recons = geometry.fbp_reconstruct(exp_sinos)

            loss_s = MSE(exp_sinos, sino_batch) if minimizer else torch.tensor(0.0).to(DEVICE)
            loss_r = MSE(recons, phantom_batch) if minimizer else torch.tensor(0.0).to(DEVICE)

            adverserial_loss = torch.mean(torch.log(1-D.forward(concat_pred_inp(la_sinos, exp_sinos, known_angles)) + eps))
            # print(D.forward(concat_pred_inp(la_sinos, exp_sinos, known_angles)))
            if not minimizer:
                adverserial_loss += torch.mean(torch.log(D.forward(concat_pred_inp(la_sinos, sino_batch, known_angles)) + eps))
            # loss_gd = torch.mean(torch.log(probs_real)) + torch.mean(torch.log(1-probs_fake))

            loss = adverserial_loss + l1*loss_s + l2*loss_r

            loss.backward()
            optimizer.step()

            Lgd.append(adverserial_loss.item())            
            Ls.append(loss_s.item())
            Lr.append(loss_r.item())        

        print("Epoch:", epoch, "adverserial score:", mean(Lgd), "loss_sino:", mean(Ls), "loss recons:", mean(Lr))

    # scheduler_g.step()
    # scheduler_d.step()


save_model_checkpoint(G, optimizer, loss_s, ar, "first_gan_generator.pt")
torch.save(D.state_dict(), "first_gan_discriminator.pt")

print("Models saved")


