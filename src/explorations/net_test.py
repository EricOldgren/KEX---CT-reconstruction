import torch
from torch.utils.data import DataLoader, TensorDataset
from utils.geometry import Geometry, setup, BasicModel, DEVICE
from utils.threaded_training import multi_threaded_training
from models.fbpnet import FBPNet
from models.doublenet import DoubleNet

ANGLE_RATIOS = [0.5]#, 0.85, 0.9, 0.95, 1.0]
EPOPCHS =      [60] #, 100, 100,  100, 60]
TRAINED = {}

scale_factor = 10.0 #10 #output is zero all the time, a large scale factor will increase the penalty of that output

if __name__ == '__main__':

    for ar, n_epochs in zip(ANGLE_RATIOS, EPOPCHS):
        geometry = Geometry(ar, 200, 100, reco_shape=(128, 128))
        train_sinos, train_y, test_sinos, test_y = setup(geometry, num_to_generate=200, train_ratio=0.99, use_realistic=False, data_path="data/kits_phantoms_256.pt")
        model = FBPNet(geometry, use_smooth_filters=False, n_fbps=2)
        # model = DoubleNet(geometry, use_smooth_filters=True)

        #dataset = list(zip(train_sinos.to(DEVICE, non_blocking=True), train_y.to(DEVICE, non_blocking=True)))
        dataset = TensorDataset(train_sinos.cpu()*scale_factor, train_y.cpu()*scale_factor)

        multi_threaded_training(model, dataset, n_epochs=n_epochs, batch_size=20, lr=0.003, regularisation_lambda=0.01, num_threads=10)

        TRAINED[ar] = model
        model.visualize_output(test_sinos, test_y, output_location="show")
        torch.save(model.state_dict(), "testing.pt")
        print("Done")
    

