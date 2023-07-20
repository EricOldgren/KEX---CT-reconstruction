import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.parallel_geometry import ParallelGeometry, setup, BasicModel
from models.fbpnet import FBPNet
import random
import time

ANGLE_RATIOS = [0.5]#[0.8, 0.85, 0.9, 0.95, 1.0]
EPOPCHS =      [3]#[100, 100, 100,  100, 60]
TRAINED = {}
LAMBDA  = 0.01 #regularization parameter

NUM_THREADS = 8



def train(model, dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = lambda diff : torch.mean(diff*diff)

    for sinos, y in dataloader:
        optimizer.zero_grad()

        out = model(sinos)
        loss = loss_fn(out-y)
        loss += model.regularization_term()*LAMBDA

        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    A = time.time()
    geometry = ParallelGeometry(0.5, 300, 150)
    train_sinos, train_y, test_sinos, test_y = setup(geometry, num_to_generate=400,use_realistic=True,data_path="data/kits_phantoms_256.pt")
    model = FBPNet(geometry, n_fbps=5)
    model.share_memory()

    dataset = list(zip(train_sinos, train_y))
    B = time.time()
    print(f"Training starts after {B-A}s")
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    train(model, dataloader)

    C = time.time()
    print(f"Done after another {C-B}s")