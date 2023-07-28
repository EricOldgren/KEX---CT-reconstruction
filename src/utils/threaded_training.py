import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import numpy as np

from utils.geometries import DEVICE
from utils.moments import SinoMoments
from models.modelbase import FBPModelBase
from models.expnet import FNOExtrapolatingBP as ExpBP

# mp.set_sharing_strategy("file_system")

def train_thread(model: FBPModelBase, dataloader: DataLoader, validation_data: tuple[torch.Tensor], n_epochs: int = 100, lr=0.001, is_display_thread = False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = lambda diff : torch.mean(diff*diff)

    for epoch in range(n_epochs):
        if epoch % 10 == 0 and is_display_thread:
            model.visualize_output(sinos, y, output_location="files")
        batch_losses = []
        for sinos, y in dataloader:
            optimizer.zero_grad()
            sinos, y = sinos.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

            out = model(sinos)
            loss = loss_fn(out-y)

            batch_losses.append(loss.item())

            loss.backward()
            optimizer.step()
        if is_display_thread:
            print(f"Epoch: {epoch}, Training loss: {np.mean(batch_losses)}")
    del optimizer
    del loss_fn; del batch_losses
    del sinos; del y

def train_exp_thread(model: ExpBP, sm : SinoMoments, dataloader: DataLoader, validation_data: tuple[torch.Tensor], n_epochs: int = 100, lr = 0.001, is_display_thread = False):
    """
        Function to run the training loop for a model which extrapolates the sinograms in a subbprocess.
        Too many parameters to pass everything in here. Edit function manually instead. This function is not usefull on GPU so no need to use it in colab.
    """
    n_moments = sm.n_moments
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_fn = lambda diff : torch.mean(diff**2) #maybe add grad_sq also
    val_sinos, val_y, full_val_sinos = validation_data

    for epoch in range(n_epochs):
        if epoch % 10 == 0 and is_display_thread:
            model.visualize_output(val_sinos, val_y, full_val_sinos, output_location="files", dirname="data")
        batch_mses, batch_mse_sinos, batch_mse_moments = [], [], []
        for sinos, y, full_sinos in dataloader:
            optimizer.zero_grad()
            exp_sinos = model.extrapolate(sinos)
            mse_sino = mse_fn(full_sinos - exp_sinos); batch_mse_sinos.append(mse_sino.item())

            moms = [sm.get_moment(exp_sinos, ni) for ni in range(n_moments)]
            proj_moms = (sm.project_moment(mom, ni) for ni, mom in enumerate(moms))
            reg_mom = sum(mse_fn(mom-proj_mom) for mom, proj_mom in zip(moms, proj_moms)) / len(moms); batch_mse_moments.append(reg_mom.item())

            recon = model.fbp(exp_sinos)
            mse = mse_fn(recon - y); batch_mses.append(mse.item())

            loss = mse + mse_sino #+ 0.001 * reg_mom
            loss = loss + sum(torch.sum(torch.abs(p*p)) for p in model.parameters()) * 1e-7
            loss.backward()
            optimizer.step()
        if is_display_thread:
            print(f"Epoch {epoch} mse {np.mean(batch_mses)} sino-mse {np.mean(batch_mse_sinos)} moment mse {np.mean(batch_mse_moments)}")
    
    del sm
    del optimizer
    del mse_fn; del batch_mses; del batch_mse_sinos; del batch_mse_moments
    del val_sinos; del val_y; del full_val_sinos
    del sinos; del y; del full_sinos

def multi_threaded_training(model: torch.nn.Module, dataset: Dataset, validation_data: tuple[torch.Tensor], n_epochs=100, batch_size=32, lr=0.001, num_threads=8, exp_model=False):
    """
        Creates specified number of subprocesses via the 'torch.multiprocessing' submodule. The purpose is to make the training loop faster.

        Every subprocess runs its own training loop that processes a subset of the training data. 

        Note: If using this function, it appears code that is not in a "if __name__ == '__main__'" - block is run once for every thread!!!
    """

    model.share_memory()
    n_moments = 5
    sm = SinoMoments(model.extended_geometry, n_moments=n_moments)

    processes = []

    for rank in range(num_threads):
        dataloader = DataLoader(
                        dataset=dataset,
                        batch_size=batch_size,
                        #pin_memory=True,
                        sampler=DistributedSampler(
                            dataset=dataset,
                            num_replicas=num_threads,
                            rank=rank
                        )
        )
        if exp_model:
            p = mp.Process(target=train_exp_thread, args=(model, sm, dataloader, validation_data, n_epochs, lr, rank==0))
        else:
            p = mp.Process(target=train_thread, args=(model, dataloader, validation_data, n_epochs, lr, rank==0))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    print("Done training and subprocesses are gathered!")