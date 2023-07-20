import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from utils.parallel_geometry import ParallelGeometry, setup, BasicModel
from models.fbpnet import FBPNet
import random
import time

ANGLE_RATIOS = [0.5]#[0.8, 0.85, 0.9, 0.95, 1.0]
EPOPCHS =      [3]#[100, 100, 100,  100, 60]
TRAINED = {}
LAMBDA  = 0.01 #regularization parameter

NUM_THREADS = 4


def train(model, dataloader, display_loss = False):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = lambda diff : torch.mean(diff*diff)

    for epoch in range(100):
        batch_losses = []
        for sinos, y in dataloader:
            optimizer.zero_grad()

            out = model(sinos)
            loss = loss_fn(out-y)
            if display_loss:
                batch_losses.append(loss.item())

            # loss += model.regularization_term()*LAMBDA

            loss.backward()
            optimizer.step()
        if display_loss:
            print(f"Epoch: {epoch}, Training loss: {sum(batch_losses) / len(batch_losses)}")

def test_eval(model, test_sinos, test_y):
    with torch.no_grad():
        out = model(test_sinos)
        loss = torch.mean((out-test_y)**2)
    print(f"Validation Loss is {loss.item()}")

def mp_epoch(model, dataset, batch_size=32, display_loss = False):

        processes = []
        for rank in range(NUM_THREADS):
            dataloader = DataLoader(dataset, batch_size=batch_size, sampler=DistributedSampler(
                            dataset=dataset,
                            num_replicas=NUM_THREADS,
                            rank=rank)
                        )
            p = mp.Process(target=train, args=(model, dataloader, (rank == 0 and display_loss)))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

if __name__ == '__main__':
    A = time.time()
    geometry = ParallelGeometry(0.5, 300, 150)
    train_sinos, train_y, test_sinos, test_y = setup(geometry, num_to_generate=80,use_realistic=False,data_path="data/kits_phantoms_256.pt")
    model = FBPNet(geometry, n_fbps=5)
    model.share_memory()

    dataset = list(zip(train_sinos, train_y))
    B = time.time()
    print(f"Done generating after {B-A}s")
    
    print(f"Training starts after {B-A}s")
    n_epochs = 100

    processes = []
    for rank in range(NUM_THREADS):
        dataloader = DataLoader(dataset, batch_size=32, sampler=DistributedSampler(
                        dataset=dataset,
                        num_replicas=NUM_THREADS,
                        rank=rank)
                    )
        
        p = mp.Process(target=train, args=(model, dataloader, rank==0))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    C = time.time()
    print(f"Done after another {C-B}s")