import torch

from utils.geometry import Geometry

def test_fourier_transforms():
    geometry = Geometry(1.0, 150, 100)

    x = torch.randn((geometry.t_size)).to(device="cuda")

    xhat = geometry.fourier_transform(x)

    recovered = geometry.inverse_fourier_transform(xhat)

    assert (torch.abs(x-recovered) < 1e-4).all(), "Earrrghhh"


if __name__ == '__main__':
    test_fourier_transforms()