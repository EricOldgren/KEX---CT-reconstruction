import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.geometry import Geometry, setup, BasicModel
import random
from models.analyticmodels import RamLak

ANGLE_RATIOS = [1.0, .8]
EPOPCHS =      [100, 60]
TRAINED = {}


for ar in ANGLE_RATIOS:

    #For maximazing omega - phi_size ~ pi * t_size / 2
    (test_sinos, test_y, _, _), geometry = setup(ar, phi_size=450, t_size=300, train_ratio=1.0, num_samples=10)
    analytic = RamLak(geometry)

    analytic.visualize_output(test_sinos, test_y, lambda diff : torch.mean(diff*diff))
    plt.show()

