import torch
from torch.utils.data import DataLoader
from utils.geometry import Geometry, setup, BasicModel
from models.fbpnet import FBPNet

ANGLE_RATIOS = [0.8, 0.85, 0.9, 0.95, 1.0]
EPOPCHS =      [100, 100, 100,  100, 60]
TRAINED = {}

for ar, n_epochs in zip(ANGLE_RATIOS, EPOPCHS):
    (train_sinos, train_y, test_sinos, test_y), geometry = setup(ar, phi_size=300, t_size=100, num_samples=10)
    model = FBPNet(geometry)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = lambda diff : torch.mean(diff*diff)

    dataloader = DataLoader(list(zip(train_sinos, train_y)), batch_size=25, shuffle=True)

    for epoch in range(n_epochs):
        if epoch % 10 == 0:
            model.visualize_output(test_sinos, test_y, loss_fn)
        for sinos, y in dataloader:
            out = model(sinos)

            loss = loss_fn(out - y) #+ abs(sum(out[int(geometry.omega):]))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"epoch {epoch} loss: {loss.item()}")
    
    TRAINED[ar] = model
