from typing import Literal
import torch
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

from geometries import FBPGeometryBase

class FBPModelBase(torch.nn.Module, ABC):

    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def get_extrapolated_sinos(self,  sinos: torch.Tensor):
        """Get sinos extrapolated with model - required for evaluation plotting
        """
    
    @abstractmethod
    def get_extrapolated_filtered_sinos(self, sinos: torch.Tensor):
        """Get filtered full sinos - required evaluation for plotting
        """



def evaluate_batches(pred: torch.Tensor, gt: torch.Tensor, ind: int, title: str):
    "return fig, mse"
    N, _, _  = pred.shape
    assert 0 <= ind < N
    mse = torch.mean((pred-gt)**2)
    fig, _ = plt.subplots(1,2)
    plt.subplot(121)
    plt.imshow(pred[ind].cpu())
    plt.title(title + " - predicted")
    plt.subplot(122)
    plt.imshow(gt[ind].cpu())
    plt.title(title + " - gt")

    return fig, mse

def plot_model_progress(model: FBPModelBase, geometry: FBPGeometryBase, cropped_sinos: torch.Tensor, full_sinos: torch.Tensor, phantoms: torch.Tensor):
    """
        Print mses and plot reconstruction samples for model. This will display, sinogram extrappolation, sinogram filtering and reconstruction
    """
    N, _, _ = cropped_sinos.shape

    full_filtered_sinos = geometry.inverse_fourier_transform(geometry.fourier_transform(full_sinos*geometry.jacobian_det)*geometry.ram_lak_filter())
    with torch.no_grad():
        exp_sinos = model.get_extrapolated_sinos(cropped_sinos)
        filtered_sinos = model.get_extrapolated_filtered_sinos(cropped_sinos)
        recons = model(cropped_sinos)
    
    disp_ind = torch.randint(0, N, (1,)).item()
    sin_fig, sin_mse = evaluate_batches(exp_sinos, full_sinos, disp_ind, "sinogram")
    filtered_sin_fig, filtered_sin_mse = evaluate_batches(filtered_sinos, full_filtered_sinos, disp_ind, "filtered sinograms")
    recon_fig, recon_mse = evaluate_batches(recons, phantoms, disp_ind, "reconstruction")

    print("sinogram mse:", sin_mse)
    print("filterd sinogram mse: ", filtered_sin_mse)
    print("reconstruction mse: ", recon_mse)

    sin_fig.show()
    filtered_sin_fig.show()
    recon_fig.show()
    plt.show()