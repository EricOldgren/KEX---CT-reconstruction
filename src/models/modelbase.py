from typing import Literal
import torch
import random
from abc import ABC, abstractmethod


class FBPModelBase(ABC):
    
    @abstractmethod
    def get_extrapolated_sinos(self,  sinos: torch.Tensor):
        """Get sinos extrapolated with model - used for plotting
        """
    
    @abstractmethod
    def get_extrapolated_filtered_sinos(self, sinos: torch.Tensor):
        """Get filtered full sinos - used for plotting
        """

def evaluate_model(model: FBPModelBase, cropped_sinos: torch.Tensor, full_sinos: torch.Tensor, gt_y: torch.Tensor,  output_location: Literal["files", "show"] = "files", dirname = "data", prettify_output = True):
    """Print performance and visualize som sample output from model to evaluate model

    Args:
        model (FBPModelBase): _description_
        cropped_sinos (torch.Tensor): _description_
        full_sinos (torch.Tensor): _description_
        gt_data (torch.Tensor): _description_
    """
    N, h, w = cropped_sinos.shape
    loss_fn = lambda diff : torch.mean(diff**2)
    assert full_sinos == (N, h, w)
    ind = random.randint(0, N-1)
    with torch.no_grad():
        test_out = model(cropped_sinos)  
    loss = loss_fn(gt_y-test_out)
    print()
    print(f"Evaluating current model state, validation loss: {loss.item()}. Displayiing sample nr {ind}: ")
    sample_sino, sample_y, sample_out = cropped_sinos[ind].to("cpu"), gt_y[ind].to("cpu"), test_out[ind].to("cpu")
    if prettify_output:
        sample_out = torch.minimum(sample_out, torch.ones(sample_out.shape, device="cpu")*torch.max(sample_y))

    if self.reconstructionfig is None:
        self.reconstructionfig, (ax_gt, ax_recon) = plt.subplots(1,2)
    else:
        ax_gt, ax_recon = self.reconstructionfig.get_axes()

    self.reconstructionfig.suptitle(f"Output. Averaged MSE {loss.item()}")
    ax_gt.imshow(sample_y)
    ax_gt.set_title("Real Data")
    ax_recon.imshow(sample_out)
    ax_recon.set_title("Reconstruction")
    
    if self.plotkernels: self.draw_kernels()

    if output_location == "files":
        self.reconstructionfig.savefig(os.path.join(dirname, "output-while-running"))
        if self.plotkernels: self.kernelfig.savefig(f"{os.path.join(dirname, 'kernels-while-running')}")
        self.reconstructionfig = None
        self.kernelfig = None
        print("Updated plots saved as files")
    elif output_location == "show":
        self.reconstructionfig.show()
        if self.plotkernels: self.kernelfig.show()
        plt.show()
        self.reconstructionfig = None
        self.kernelfig = None
    else:
        raise ValueError(f"Invalid output location {output_location}")    