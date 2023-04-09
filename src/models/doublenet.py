import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.geometry import Geometry
from models.fbpnet import FBPNet
from math  import ceil
import odl.contrib.torch as odl_torch
import random

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class DoubleNet(nn.Module):

    reconstructionfig: Figure = None

    def __init__(self, geometry: Geometry, use_smooth_filters: bool = False, smooth_basis = None, **kwargs):

        super().__init__(**kwargs)
        if smooth_basis is not None: raise NotImplementedError()
        
        self.geometry1 = geometry # reco space could in theory be different :/, probably no real benefit from it

        self.net1a = FBPNet(self.geometry1, 2, use_smooth_filters=use_smooth_filters)
        self.net1b = FBPNet(self.geometry1, 2, use_smooth_filters=use_smooth_filters)

        full_phi_size, full_t_size,  = ceil(geometry.phi_size * 1.0 / geometry.ar), ceil(geometry.t_size * 1.0 / geometry.ar)
        self.geometry2 = Geometry(1.0, full_phi_size, full_t_size, reco_shape=geometry.reco_space.shape)        
        self.ray_layer = odl_torch.OperatorModule(self.geometry2.ray) #needs to have same reco space as geometry 1!!! OK now since reco space 1 is the same as the final

        self.net2 = FBPNet(self.geometry2, 4, use_smooth_filters=use_smooth_filters)
    
    def forward(self, limited_ar_sinos):

        recona = self.net1a(limited_ar_sinos)
        reconb = self.net1b(limited_ar_sinos)

        full_ar_sinosa = F.relu(self.ray_layer(recona))
        full_ar_sinosb = F.relu(self.ray_layer(reconb))

        full_ar_sinos = full_ar_sinosa + full_ar_sinosb

        return self.net2(full_ar_sinos)

    def regularization_term(self):
        return  self.net1a.regularization_term() + self.net1b.regularization_term() + self.net2.regularization_term()

    def visualize_output(self, test_sinos, test_y, loss_fn = lambda diff : torch.mean(diff*diff), output_location = "files"):

            ind = random.randint(0, test_sinos.shape[0]-1)
            with torch.no_grad():
                test_out = self.forward(test_sinos)  
            loss = loss_fn(test_y-test_out)
            print()
            print(f"Evaluating current model state, validation loss: {loss.item()} using angle ratio: {self.geometry1.ar}. Displayiing sample nr {ind}: ")
            sample_sino, sample_y, sample_out = test_sinos[ind].to("cpu"), test_y[ind].to("cpu"), test_out[ind].to("cpu")

            if self.reconstructionfig is None:
                self.reconstructionfig, (ax_gt, ax_recon) = plt.subplots(1,2)
            else:
                ax_gt, ax_recon = self.reconstructionfig.get_axes()

            ax_gt.imshow(sample_y)
            ax_gt.set_title("Real Data")
            ax_recon.imshow(sample_out)
            ax_recon.set_title("Reconstruction")

            if output_location == "files":
                self.reconstructionfig.savefig("data/output-while-running")
                print("Updated plots saved as files")
            else:
                self.reconstructionfig.show()
                plt.show()
                self.reconstructionfig = None