import torch
from typing import Literal, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass
import matplotlib.pyplot as plt

from utils.tools import PathType
from geometries import FBPGeometryBase, AVAILABLE_FBP_GEOMETRIES

class FBPModelBase(torch.nn.Module, ABC):

    geometry: FBPGeometryBase
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def get_extrapolated_sinos(self,  sinos: torch.Tensor, known_angles: torch.Tensor, angles_out: torch.Tensor):
        """Get sinos extrapolated with model - required for evaluation plotting
        """
    
    @abstractmethod
    def get_extrapolated_filtered_sinos(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out: torch.Tensor):
        """Get filtered full sinos - required evaluation for plotting
        """
    
    @abstractmethod
    def get_init_torch_args(self):
        """Get args used in init method that are torch saveable in a state_dict, this cannot include `model.geometry`.
            Necessary to reload model after saving a model.
        """
    
    @abstractmethod
    def forward(self, sinos: torch.Tensor, known_angles: torch.Tensor, angles_out: torch.Tensor):
        ...



def evaluate_batches(pred: torch.Tensor, gt: torch.Tensor, ind: int, title: str):
    "return fig, mse"
    N, _, _  = pred.shape
    M = torch.max(torch.abs(gt))
    vmin, vmax  = max(torch.min(pred), -M), min(torch.max(pred), M)
    assert 0 <= ind < N
    mse = torch.mean((pred-gt)**2)
    fig, _ = plt.subplots(1,2)
    fig.suptitle(f"{title}, MSE: {mse:.4f}")
    plt.subplot(121)
    plt.imshow(pred[ind].cpu(), vmin=vmin, vmax=vmax)
    plt.title("predicted")
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(gt[ind].cpu(), vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title("gt")

    return fig, mse

def plot_model_progress(model: FBPModelBase, full_sinos: torch.Tensor, known_angles: torch.Tensor, phantoms: torch.Tensor,out_angles: torch.Tensor = None, disp_ind: int = 0, model_name: str = None, force_show=False):
    """
        Print mses and plot reconstruction samples for model.
        This will display: sinogram extrappolation, sinogram filtering and reconstruction
    """
    N, _, _ = full_sinos.shape

    geometry = model.geometry
    full_filtered_sinos = geometry.inverse_fourier_transform(geometry.fourier_transform(full_sinos*geometry.jacobian_det)*geometry.ram_lak_filter())
    cropped_sinos = full_sinos * 0
    cropped_sinos[:, known_angles] = full_sinos[:, known_angles]
    with torch.no_grad():
        exp_sinos = model.get_extrapolated_sinos(cropped_sinos, known_angles, out_angles)
        filtered_sinos = model.get_extrapolated_filtered_sinos(cropped_sinos, known_angles, out_angles)
        recons = model(cropped_sinos, known_angles, out_angles)

    if model_name is None:
        model_name = type(model).__name__
    sin_fig, sin_mse = evaluate_batches(exp_sinos, full_sinos, disp_ind, title=f"{model_name} - sinograms")
    filtered_sin_fig, filtered_sin_mse = evaluate_batches(filtered_sinos, full_filtered_sinos, disp_ind, title=f"{model_name} - filtered sinograms")
    recon_fig, recon_mse = evaluate_batches(recons, phantoms, disp_ind, title=f"{model_name} - reconstructions")

    print("="*40)
    print(model_name)
    print("sinogram mse:", sin_mse)
    print("filterd sinogram mse: ", filtered_sin_mse)
    print("reconstruction mse: ", recon_mse)

    sin_fig.show()
    filtered_sin_fig.show()
    recon_fig.show()
    if force_show:
        plt.show()

@dataclass
class TrainingCheckPoint:
    model: FBPModelBase
    geometry: FBPGeometryBase
    optimizer: torch.optim.Optimizer
    loss: torch.Tensor
    angle_ratio: float


pytorch_optimizers = [
    torch.optim.Adadelta,
    torch.optim.Adagrad,
    torch.optim.Adam,
    torch.optim.AdamW,
    torch.optim.SparseAdam,
    torch.optim.Adamax,
    torch.optim.ASGD,
    torch.optim.LBFGS,
    torch.optim.NAdam,
    torch.optim.RAdam,
    torch.optim.RMSprop,
    torch.optim.Rprop,
    torch.optim.SGD
]
pytorch_optimizer_dict = {opt.__name__: opt for opt in pytorch_optimizers}
fbp_geometry_dict = {g.__name__: g for g in AVAILABLE_FBP_GEOMETRIES}
def save_model_checkpoint(model: FBPModelBase, optimizer: torch.optim.Optimizer, loss: torch.Tensor, angle_ratio: float, path: PathType):
    """Save model and training data.

    Args:
        model (FBPModelBase): _description_
        optimizer (torch.optim.Optimizer): _description_
        loss (torch.Tensor): _description_
        angle_ratio (float): _description_
        path (PathType): _description_
    """

    if not type(optimizer).__name__ in pytorch_optimizer_dict:
        print("Optimizer class is not recognized, resuming training from this checkpoint may not work as expected!")
    if not type(model.geometry).__name__ in fbp_geometry_dict:
        print("Geometry unrecognized, loading this model may not work!")
        
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_args": model.get_init_torch_args(),
        "geometry_args": model.geometry.get_init_args(),
        "geometry_class_name": type(model.geometry).__name__,
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_class_name": type(optimizer).__name__,
        "loss": loss,
        "angle_ratio": angle_ratio
    }, path)


def load_model_checkpoint(path: PathType, ModelClass: Type[FBPModelBase]):
    state_dict = torch.load(path)
    
    model_state_dict = state_dict["model_state_dict"]
    model_args = state_dict["model_args"]
    geometry_args = state_dict["geometry_args"]
    geometry_class_name = state_dict["geometry_class_name"]
    optimizer_state_dict = state_dict["optimizer_state_dict"]
    optimizer_class_name = state_dict["optimizer_class_name"]
    loss = state_dict["loss"]
    angle_ratio = state_dict["angle_ratio"]

    assert geometry_class_name in fbp_geometry_dict, f"unrecognized geometry type, {geometry_class_name}"
    geometry = fbp_geometry_dict[geometry_class_name](*geometry_args)

    model = ModelClass(geometry, *model_args)
    model.load_state_dict(model_state_dict)
    
    try:
        optimizer: torch.optim.Optimizer = pytorch_optimizer_dict[optimizer_class_name](model.parameters(), lr=1.0)
        optimizer.load_state_dict(optimizer_state_dict)
    except:
        print("Couldn't load optimizer.")
        optimizer = None
    
    return TrainingCheckPoint(model, geometry, optimizer, loss, angle_ratio)

