import torch
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d=1
t_size=1
a=1
fourier_domain: torch.Tensor = 2*np.pi * torch.fft.rfftfreq(t_size, d).to(DEVICE)
print(-1j*a*fourier_domain)
print(torch.__version__)
print(torch.version.cuda)
#-1j*a*fourier_domain
torch.cos(a*fourier_domain)+1j*torch.sin(a*fourier_domain)