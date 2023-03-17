import torch
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np



def plot_them():
    a = -8
    b = 8
    N = 1000
    d = (b-a) / N
    x = torch.linspace(a, b, N)
    f = torch.exp(-x*x/2)

    dft = torch.fft.rfft(f) #Discrete fourier transform
    print(f.shape)
    print(dft.shape)

    w = 2*np.pi*torch.fft.rfftfreq(N, d=d) #omega interval
    dw = w[1]-w[0]
    fhat = d*torch.exp(-1j*a*w)*dft
    print("Angular velocity step, from torch and by hand:")
    print(dw)
    print(2*np.pi/(N*d))
    ha = sqrt(2*torch.pi)*torch.exp(-w*w/2) #fourier transform calculated by hand
    

    plt.subplot(121)
    plt.title("Function in time domain")
    plt.plot(x, f)
    plt.subplot(122)
    plt.title("Transforms in freq domain")
    plt.plot(w, fhat, label='numeric')
    plt.plot(w, ha, label="analytic - from wolfram")
    plt.legend()

    plt.show()

plot_them()

