import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.parallel_geometry import ParallelGeometry, setup, BasicModel
import random
from models.analyticmodels import RamLak, ramlak_filter
import odl.contrib.torch as odl_torch
from models.modelbase import ChainedModels
from models.fbps import FBP

ANGLE_RATIOS = [0.5]


for ar in ANGLE_RATIOS:

    #For maximazing omega - phi_size ~ pi * t_size / 2
    geometry = ParallelGeometry(1.0, phi_size=450, t_size=300)
    geom2 = ParallelGeometry(1.0, 300, 150)
    ray = odl_torch.OperatorModule(geom2.ray)
    (test_sinos, test_y, _, _) = setup(geometry, train_ratio=0.03, num_to_generate=0, use_realistic=True, data_path="data/kits_phantoms_256.pt")
    analytic = RamLak(geometry) #FBP(geometry, initial_kernel=ramlak_filter(geometry) / 0.5, trainable_kernel=False)
    v2 = RamLak(geom2)

    # Y = test_y[0]
    # X = test_sinos[0]
    # Z = analytic(X[None])[0]
    # Z = torch.minimum(Z, torch.ones(Z.shape, dtype=Z.dtype)*torch.max(Y))

    # plt.subplot(121)
    # plt.imshow(Y)
    # plt.subplot(122)
    # plt.imshow(Z)
    # plt.show()

    better = ray(F.relu(analytic(test_sinos)))

    # model = ChainedModels([analytic, v2])
    # model.visualize_output(test_sinos, test_y, output_location="show")
    analytic.visualize_output(test_sinos, test_y, output_location="show")

    # v2.visualize_output(better, test_y, lambda diff : torch.mean(diff*diff))
