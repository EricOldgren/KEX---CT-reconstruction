import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.geometry import Geometry, setup, BasicModel
from utils.analyticfilter import analytic_model
import random

ANGLE_RATIOS = [1.0, .8]
EPOPCHS =      [100, 60]
TRAINED = {}

for ar in ANGLE_RATIOS:

    (test_sinos, test_y, _, _), geometry = setup(ar, phi_size=240, t_size=400, train_ratio=1.0, num_samples=30)

    analytic = analytic_model(geometry)

    analytic.visualize_output(test_sinos, test_y, lambda diff : torch.mean(diff*diff))
    plt.show()

