from typing import Literal, Type
import torch
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

from geometries import FBPGeometryBase, FlatFanBeamGeometry, ParallelGeometry

class FBPModelBase(torch.nn.Module, ABC):

    geometry: FBPGeometryBase
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
    
    @abstractmethod
    def get_init_torch_args(self):
        """Get args used in init method that are torch saveable in a state_dict, this cannot include `model.geometry`.
            Necessary to reload model after saving a model.
        """



def evaluate_batches(pred: torch.Tensor, gt: torch.Tensor, ind: int, title: str):
    "return fig, mse"
    N, _, _  = pred.shape
    M = torch.max(torch.abs(gt))
    vmin, vmax  = max(torch.min(pred), -M), min(torch.max(pred), M)
    assert 0 <= ind < N
    mse = torch.mean((pred-gt)**2)
    fig, _ = plt.subplots(1,2)
    plt.subplot(121)
    plt.imshow(pred[ind].cpu(), vmin=vmin, vmax=vmax)
    plt.title(title + " - predicted")
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(gt[ind].cpu(), vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(title + " - gt")

    return fig, mse

def plot_model_progress(model: FBPModelBase, cropped_sinos: torch.Tensor, full_sinos: torch.Tensor, phantoms: torch.Tensor, disp_ind: int = 0, force_show=False):
    """
        Print mses and plot reconstruction samples for model.
        This will display: sinogram extrappolation, sinogram filtering and reconstruction
    """
    N, _, _ = cropped_sinos.shape

    geometry = model.geometry
    full_filtered_sinos = geometry.inverse_fourier_transform(geometry.fourier_transform(full_sinos*geometry.jacobian_det)*geometry.ram_lak_filter())
    with torch.no_grad():
        exp_sinos = model.get_extrapolated_sinos(cropped_sinos)
        filtered_sinos = model.get_extrapolated_filtered_sinos(cropped_sinos)
        recons = model(cropped_sinos)
    
    sin_fig, sin_mse = evaluate_batches(exp_sinos, full_sinos, disp_ind, "sinogram")
    filtered_sin_fig, filtered_sin_mse = evaluate_batches(filtered_sinos, full_filtered_sinos, disp_ind, "filtered sinograms")
    recon_fig, recon_mse = evaluate_batches(recons, phantoms, disp_ind, "reconstruction")

    print("sinogram mse:", sin_mse)
    print("filterd sinogram mse: ", filtered_sin_mse)
    print("reconstruction mse: ", recon_mse)

    sin_fig.show()
    filtered_sin_fig.show()
    recon_fig.show()
    if force_show:
        plt.show()

def save_model_checkpoint(model: FBPModelBase, optimizer: torch.optim.Optimizer, loss: torch.Tensor, path, geometry_class: Literal["FlatFanBeam", "Parallel"], angle_ratio: float):
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_args": model.get_init_torch_args(),
        "geometry_args": model.geometry.get_init_args(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "geometry_class": geometry_class,
        "angle_ratio": angle_ratio
    }, path)

def load_model_from_checkpoint(path, ModelClass: Type[FBPModelBase]):
    state_dict = torch.load(path)
    model_state_dict = state_dict["model_state_dict"]
    model_args = state_dict["model_args"]
    geometry_args = state_dict["geometry_args"]
    geometry_class = state_dict["geometry_class"]
    if geometry_class == "FlatFanBeam":
        geometry = FlatFanBeamGeometry(*geometry_args)
    elif geometry_class == "Parallel":
        geometry = ParallelGeometry(*geometry_args)
    else:
        raise ValueError(f"Geometry class ({geometry_class}) in state_dict is invalid!")
    
    model = ModelClass(geometry, *model_args)
    model.load_state_dict(model_state_dict)

    return model

