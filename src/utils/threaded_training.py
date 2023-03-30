import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler, Dataset

mp.set_sharing_strategy("file_system")

def train_thread(model: torch.nn.Module, dataloader: DataLoader, n_epochs: int = 100, lr=0.01, regularisation_lambda: float = 1e-3, display_loss: bool = False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = lambda diff : torch.mean(diff*diff)

    for epoch in range(n_epochs):
        batch_losses = []
        for sinos, y in dataloader:
            optimizer.zero_grad()

            out = model(sinos)
            loss = loss_fn(out-y)
            if display_loss:
                batch_losses.append(loss.item())

            loss += model.regularization_term()*regularisation_lambda

            loss.backward()
            optimizer.step()
        if display_loss:
            print(f"Epoch: {epoch}, Training loss: {sum(batch_losses) / len(batch_losses)}")
    del optimizer
    del loss_fn; del batch_losses
    del sinos; del y

def multi_threaded_training(model: torch.nn.Module, dataset: Dataset, n_epochs=100, batch_size=32, lr=0.01, regularisation_lambda=1e-3, num_threads=8):
    """
        Creates specified number of subprocesses via the 'torch.multiprocessing' submodule. The purpose is to make the training loop faster.

        Every subprocess runs its own training loop that processes a subset of the training data. 

        Note: If using this function, it appears code that is not in a "if __name__ == '__main__'" - block is run once for every thread!!!
    """

    model.share_memory()

    processes = []

    for rank in range(num_threads):
        dataloader = DataLoader(
                        dataset=dataset,
                        batch_size=batch_size,
                        sampler=DistributedSampler(
                            dataset=dataset,
                            num_replicas=num_threads,
                            rank=rank
                        )
        )

        p = mp.Process(target=train_thread, args=(model, dataloader, n_epochs, lr, regularisation_lambda, rank==0))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    print("Done training and subprocesses are gathered!")