import torch
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np


from geometries import ParallelGeometry, DEVICE

def test_fourier_transforms():
    geometry = ParallelGeometry(1.0, 150, 100)

    x = torch.randn((1,1,geometry.Nt)).to(device=DEVICE)

    xhat = geometry.fourier_transform(x, padding=False)
    recovered = geometry.inverse_fourier_transform(xhat, padding=False)
    assert (torch.abs(x-recovered) < 1e-4).all(), "Earrrghhh"

    xhat = geometry.fourier_transform(x, padding=True)
    recovered = geometry.inverse_fourier_transform(xhat, padding=True)
    assert (torch.abs(x-recovered) < 1e-4).all(), "Eauurghhh"

    print("Geometry fourier and inv fourier passed")

def plot_them():
    print("Plottiing FTs (not from geometry)")
    a = -8
    b = 8
    N = 10000
    d = (b-a) / N
    x = torch.linspace(a, b, N)
    f = torch.exp(-x*x/2)

    dft = torch.fft.rfft(f) #Discrete fourier transform
    # print(f.shape)
    # print(dft.shape)

    w = 2*np.pi*torch.fft.rfftfreq(N, d=d) #omega interval
    dw = w[1]-w[0]
    fhat = d*torch.exp(-1j*a*w)*dft #scaled DFT to match FT
    assert np.abs(dw - 2*np.pi/(N*d)) < 1e-4
    assert torch.mean((w - torch.linspace(0, (N//2 + 1)*dw, N //2 + 1))**2) / torch.max(w) < 1e-4
    # print("Angular velocity step, from torch and by hand:")
    # print(dw)
    # print(2*np.pi/(N*d))
    ha = sqrt(2*torch.pi)*torch.exp(-w*w/2).to(dtype=torch.complex64) #fourier transform calculated analytically
    
    assert (torch.abs(ha-fhat) < 0.01).all() #difference is at max ~0.001 :/

    plt.subplot(121)
    plt.title("Function in time domain")
    plt.plot(x, f)
    plt.subplot(222)
    plt.title("FT - real")
    plt.plot(w, fhat.real, label='numeric')
    plt.plot(w, ha.real, label="analytic - from wolfram alpha")

    plt.subplot(224)
    plt.title("FT - imaginary")
    plt.plot(w, fhat.imag, label='numeric')
    plt.plot(w, ha.imag, label="analytic - from wolfram")
    plt.legend()

    plt.show()


if __name__ == '__main__':
    test_fourier_transforms()
    plot_them()