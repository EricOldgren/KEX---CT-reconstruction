import torch
import torch.nn.functional as F
import numpy as np
import odl
import odl.contrib.torch as odl_torch

from utils.parallel_geometry import ParallelGeometry, DEVICE
from models.analyticmodels import RamLak

geometry = ParallelGeometry(1.0, 900, 300)

MSE = lambda diff : torch.mean(diff**2)

ray_trafo = geometry.ray
ray_layer = odl_torch.OperatorModule(ray_trafo)

fourier = odl.trafos.FourierTransform(ray_trafo.range, axes=[1])
ramp_function = fourier.range.element(lambda x: np.abs(x[1]) / (2 * np.pi))
ramp_filter = fourier.inverse * ramp_function * fourier
# Create filtered back-projection by composing the back-projection (adjoint)
# with the ramp filter.
fbp_odl_by_hand = ray_trafo.adjoint * ramp_filter
fbp_odl_by_hand_l = odl_torch.OperatorModule(fbp_odl_by_hand)

fbp_odl = odl.tomo.fbp_op(ray_trafo)
fbp_odl_l = odl_torch.OperatorModule(fbp_odl)

ramlak = RamLak(geometry)

models = [ramlak, fbp_odl_l, fbp_odl_by_hand_l, ]
names = ["ramlak", "fbp odl", "fpb odl by hand"]

read_data: torch.Tensor = torch.load("data/kits_phantoms_256.pt").moveaxis(0,1).to(DEVICE)
read_data = torch.concat([read_data[1], read_data[0], read_data[2]])
read_data = read_data[:50]
read_data /= torch.max(torch.max(read_data, dim=-1).values, dim=-1).values[:, None, None]

test_sinos = ray_layer(read_data)
test_y = read_data

ramlak.visualize_output(test_sinos, test_y, output_location="show")
print("Evaluation started...")

for m, nm in zip(models, names):
    out = m(test_sinos)

    print(f"MSE from {nm} : {MSE(test_y-out)}")