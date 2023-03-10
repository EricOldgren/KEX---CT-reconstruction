import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.geometry import Geometry, setup, BasicModel
from models.fbpnet import FBPNet
import random

ANGLE_RATIOS = [0.8, 0.85, 0.9, 0.95, 1.0]
EPOPCHS =      [100, 100, 100,  100, 60]
TRAINED = {}
LAMBDA  = 0.01 #regularization parameter

for ar, n_epochs in zip(ANGLE_RATIOS, EPOPCHS):
    geometry = Geometry(ar, 300, 150)
    (train_sinos, train_y, test_sinos, test_y) = setup(geometry, num_samples=10)
    model = FBPNet(geometry, n_fbps=8)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = lambda diff : torch.mean(diff*diff)

    dataloader = DataLoader(list(zip(train_sinos, train_y)), batch_size=25, shuffle=True)

    for epoch in range(n_epochs):
        if epoch % 10 == 0:
            model.visualize_output(test_sinos, test_y, loss_fn)
        for sinos, y in dataloader:
            out = model(sinos)

            loss = loss_fn(out - y) #+ abs(sum(out[int(geometry.omega):]))
            loss += model.regularization_term()*LAMBDA
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"epoch {epoch} loss: {loss.item()}")
    
    TRAINED[ar] = model

