import odl
import odl.contrib.torch as odl_torch
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Dataset
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.geometry import ParallelGeometry, setup, BasicModel
import random


def evaluate(test_sinos, test_y, model: BasicModel, loss_fn):

    ind = random.randint(0, test_sinos.shape[0]-1)
    with torch.no_grad():
        test_out = model(test_sinos)  

    loss = loss_fn(test_y-test_out)
    print()
    print(f"Evaluating current kernel, validation loss: {loss.item()} using angle ratio: {model.geometry.ar}. Displayiing sample nr {ind}: ")

    sample_sino, sample_y, sample_out = test_sinos[ind].to("cpu"), test_y[ind].to("cpu"), test_out[ind].to("cpu")
    
    plt.subplot(211)
    plt.cla()
    plt.plot(list(range(model.kernel.shape[0])), model.kernel.detach().cpu(), label="filter in frequency domain")
    plt.legend()
    plt.subplot(223)
    plt.imshow(sample_y)
    plt.title("Real data")
    plt.subplot(224)
    plt.imshow(sample_out)
    plt.title("Filtered Backprojection")
    plt.draw()

    plt.pause(0.05)

ANGLE_RATIOS = [0.8, 0.85, 0.9, 0.95, 1.0]
EPOPCHS =      [100, 100, 100,  100, 60]

for ar, n_epochs in zip(ANGLE_RATIOS, EPOPCHS):
    (train_sinos, train_y, test_sinos, test_y), geometry = setup(ar, phi_size=500, t_size=100)
    model = BasicModel(geometry)

    optimizer = torch.optim.Adam([model.kernel], lr=0.01)
    loss_fn = lambda diff : torch.mean(diff*diff)

    dataloader = DataLoader(list(zip(train_sinos, train_y)), batch_size=80, shuffle=True)

    for epoch in range(n_epochs):
        if epoch % 10 == 0:
            evaluate(test_sinos, test_y, model, loss_fn)
        for sinos, y in dataloader:
            out = model(sinos)

            loss = loss_fn(out - y)
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch} loss: {loss.item()}")


  