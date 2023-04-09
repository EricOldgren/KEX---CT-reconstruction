import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.geometry import Geometry, setup, BasicModel
from models.fbpnet import FBPNet
import random

ANGLE_RATIOS = [0.5]#, 0.8, 0.85, 0.9, 0.95, 1.0]
EPOPCHS =      [300]#, 1000, 1000, 1000,  1000, 1000]
TRAINED = {}
LAMBDA  = 0.01 #regularization parameter

for ar, n_epochs in zip(ANGLE_RATIOS, EPOPCHS):
    geometry = Geometry(ar, 300, 150) #50,40

    (train_sinos, train_y, test_sinos, test_y) = setup(geometry, num_samples=3400,use_realistic=True,data_path="data/kits_phantoms_256.pt")
    model = FBPNet(geometry, n_fbps=3)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = lambda diff : torch.mean(diff*diff)

    dataloader = DataLoader(list(zip(train_sinos, train_y)), batch_size=10, shuffle=True)

    for epoch in range(n_epochs):
        #if epoch % 10 == 0:
            #model.visualize_output(test_sinos, test_y, loss_fn, output_location="file")
        for sinos, y in dataloader:
            out = model(sinos)

            loss = loss_fn(out - y) 
            loss += model.regularization_term()*LAMBDA
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"epoch {epoch} loss: {loss.item()}")
    model.visualize_output(test_sinos, test_y, loss_fn, output_location="show")
    
    TRAINED[ar] = model

torch.save(TRAINED[0.5].state_dict(), "results\prev_res.pt")

