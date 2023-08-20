import torch
from torch.utils.data import DataLoader, TensorDataset
from geometries import FlatFanBeamGeometry, DEVICE
from models.fbps import AdaptiiveFBP as AFBP
from models.FNOBPs.fnobp import FNO_BP
from models.modelbase import plot_model_progress
from statistics import mean
import matplotlib.pyplot as plt

ar = 0.5 #angle ratio
PHANTOM_DATA = torch.stack(torch.load("data/HTC2022/HTCTrainingPhantoms.pt")).to(DEVICE)

geometry = FlatFanBeamGeometry(720, 560, 410.66, 543.74, 112, [-40,40, -40, 40], [512, 512])
SINO_DATA = geometry.project_forward(PHANTOM_DATA)

# model = AFBP(geometry)
model = FNO_BP(geometry, hidden_layers=[40,40], modes=geometry.projection_size//4)

dataset = TensorDataset(SINO_DATA, PHANTOM_DATA)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
mse_fn = lambda diff : torch.mean(diff**2)
n_epochs = 20
for epoch in range(n_epochs):
    batch_losses = []
    for sino_batch, phantom_batch in dataloader:
        optimizer.zero_grad()

        start_ind = torch.randint(0, geometry.n_projections, (1,)).item()
        la_sinos, known_beta_bool = geometry.zero_cropp_sinos(sino_batch, ar=ar, start_ind=start_ind) #known_beta_bool is True at angles where sinogram is meassured and false otherwise
        la_sinos = geometry.reflect_fill_sinos(la_sinos, known_beta_bool)
        la_sinos = geometry.rotate_sinos(la_sinos, -start_ind) #FNO needs known angles to be in the same region all the time

        filtered = model.get_extrapolated_filtered_sinos(la_sinos)
        filtered = geometry.rotate_sinos(filtered, start_ind) #rotate back
        recons = geometry.project_backward(filtered/2) #sinogram covers 360deg  - double coverage

        loss = mse_fn(phantom_batch - recons)
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.cpu().item())
    
    print("Epoch:", epoch+1, "loss is:", mean(batch_losses))


plot_model_progress(model, geometry, geometry.reflect_fill_sinos(*geometry.zero_cropp_sinos(SINO_DATA, ar, 0)), SINO_DATA, PHANTOM_DATA)


    






