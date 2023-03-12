import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import matplotlib
from utils.geometry import Geometry, BasicModel as FBP, DEVICE

relu = nn.ReLU()

class FBPNet(nn.Module):

    def __init__(self, geometry: Geometry, n_fbps = 8, **kwargs):
        "2 layer network consisting of sums of FBPs"

        super(FBPNet, self).__init__(**kwargs)

        self.geometry = geometry

        self.fbps = [(FBP(geometry), nn.Parameter(torch.randn(1).to(DEVICE))) for _ in range(n_fbps)]
        self.weights = nn.Parameter(torch.randn(n_fbps).to(DEVICE))
        self.bout = nn.Parameter(torch.randn(1).to(DEVICE))

        for i, (fbp, b) in enumerate(self.fbps):
            self.add_module(f"fbp{i}", fbp)
            self.register_parameter(f"b{i}", b)


    def forward(self, x):
        constructs = torch.stack([relu(fbp(x) + b) for fbp, b in self.fbps])

        out = torch.sum(constructs*self.weights[:, None, None, None], axis=0)

        return relu(out + self.bout)

    def regularization_term(self):
        "Returns a sum which penalizies large kernel values at large frequencies, in accordance with Nattarer's sampling Theorem"
        
        return sum(self.weights[i]*self.fbps[i][0].regularisation_term() for i in range(len(self.fbps)) )

    def convert(self, geometry: Geometry):
        m2 = FBPNet(geometry, n_fbps=len(self.fbps))
        m2.load_state_dict(self.state_dict()) #loads weights, biases and kernels -- however kernel loading may be incompatible
        for i in range(len(self.fbps)):
            m2.fbps[i][0] = self.fbps[i][0].convert(geometry) #this will raise error if incompatible
            
        return m2
        
        

    def visualize_output(self, test_sinos, test_y, loss_fn = lambda diff : torch.mean(diff*diff)):

        ind = random.randint(0, test_sinos.shape[0]-1)
        with torch.no_grad():
            test_out = self.forward(test_sinos)  

        loss = loss_fn(test_y-test_out)
        print()
        print(f"Evaluating current model state, validation loss: {loss.item()} using angle ratio: {self.geometry.ar}. Displayiing sample nr {ind}: ")

        sample_sino, sample_y, sample_out = test_sinos[ind].to("cpu"), test_y[ind].to("cpu"), test_out[ind].to("cpu")

        plt.cla()

        for i, (fbp, b) in enumerate(self.fbps):
            plt.plot(fbp.geometry.fourier_domain.cpu(), fbp.kernel.detach().cpu(), label=f"filter {i}")

        plt.legend()
        plt.figure()
        plt.subplot(121)
        plt.imshow(sample_y)
        plt.title("Real data")
        plt.subplot(122)
        plt.imshow(sample_out)
        plt.title("Filtered Backprojection")
        plt.draw()

        plt.pause(0.05)


