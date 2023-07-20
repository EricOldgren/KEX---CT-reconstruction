import torch
import odl
import odl.contrib.torch as odl_torch
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from utils.parallel_geometry import ParallelGeometry, setup
from src.models.fbpnet import FBPNet, load_fbpnet_from_dict
from models.fbps import FBP

data: torch.Tensor = torch.load("data\kits_phantoms_256.pt").moveaxis(0,1).to("cuda")
data = torch.concat([data[1], data[0], data[2]])
test_img = data[500:600]
index = 45#rnd.randrange(0,100)
test_img /= torch.max(torch.max(test_img, dim=-1).values, dim=-1).values[:, None, None]

data2: torch.Tensor = torch.load("data\constructed_data.pt").to("cuda")
data2 /= torch.max(torch.max(data2, dim=-1).values, dim=-1).values[:, None, None]

geometry = ParallelGeometry(1.0,450,300)

ray_layer = odl_torch.OperatorModule(geometry.ray)


def test():
    img1 = test_img[index].to("cpu").numpy()
    img2 = data2[index].to("cpu").numpy()
    sinos1: torch.Tensor = ray_layer(test_img)
    sinos2: torch.Tensor = ray_layer(data2)
    sino1 = sinos1[index].to("cpu").numpy()
    sino2 = sinos2[index].to("cpu").numpy()
    plt.imshow(img1,cmap="gray")
    plt.show()
    plt.imshow(img2,cmap="gray")
    plt.show()
    plt.imshow(sino1,cmap="gray")
    plt.show()
    plt.imshow(sino2,cmap="gray")
    plt.show()

    plt.axis("off")
    plt.imsave("res_img\KITS_example_img.png", img1,cmap="gray")
    plt.imsave("res_img\KITS_example_sino.png", sino1,cmap="gray",vmax=1)
    plt.imsave("res_img\constructed_example_img.png",img2,cmap="gray")
    plt.imsave("res_img\constructed_example_sino.png",sino2,cmap="gray",vmax=1)

test()