import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.geometry import Geometry, setup, BasicModel
import random
from models.analyticmodels import RamLak
import odl.contrib.torch as odl_torch
from models.modelbase import ChainedModels

ANGLE_RATIOS = [0.5]
EPOPCHS =      [100, 60]
TRAINED = {}


for ar in ANGLE_RATIOS:

    #For maximazing omega - phi_size ~ pi * t_size / 2
    geometry = Geometry(ar, phi_size=200, t_size=100)
    geom2 = Geometry(1.0, 200, 100)
    ray = odl_torch.OperatorModule(geom2.ray)
    (test_sinos, test_y, _, _) = setup(geometry, train_ratio=0.03, num_to_generate=0, use_realistic=True, data_path="data/kits_phantoms_256.pt")
    analytic = RamLak(geometry)
    v2 = RamLak(geom2)

    better = ray(F.relu(analytic(test_sinos)))

    # model = ChainedModels([analytic, v2])
    # model.visualize_output(test_sinos, test_y, output_location="show")

    v2.visualize_output(better, test_y, lambda diff : torch.mean(diff*diff))
    plt.show()

