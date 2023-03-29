import torch
from torch.utils.data import DataLoader, Dataset
from utils.geometry import Geometry, setup, BasicModel
from utils.threaded_training import multi_threaded_training
from models.fbpnet import FBPNet

ANGLE_RATIOS = [0.8, 0.85, 0.9, 0.95, 1.0]
EPOPCHS =      [100, 100, 100,  100, 60]
TRAINED = {}

if __name__ == '__main__':

    for ar, n_epochs in zip(ANGLE_RATIOS, EPOPCHS):
        geometry = Geometry(ar, 300, 150)
        train_sinos, train_y, test_sinos, test_y = setup(geometry, num_to_generate=200, use_realistic=True, data_path="data/kits_phantoms_256.pt")
        model = FBPNet(geometry)

        dataset = list(zip(train_sinos, train_y))

        multi_threaded_training(model, dataset, n_epochs=40, batch_size=32, lr=0.01, regularisation_lambda=0.01, num_threads=16)

        TRAINED[ar] = model
        torch.save(model.state_dict(), "testing.pt")
        print("Done")
    

