from typing import Literal
import torch
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

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