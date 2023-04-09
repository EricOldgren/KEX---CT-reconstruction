import torch
from torch.utils.data import DataLoader, TensorDataset
from utils.geometry import Geometry, setup, BasicModel, DEVICE
from utils.threaded_training import multi_threaded_training
from models.fbpnet import FBPNet
from models.modelbase import ChainedModels
from models.fouriernet import CrazyKernels, GeneralFBP
from models.analyticmodels import RamLak

ANGLE_RATIOS = [0.5]#, 0.85, 0.9, 0.95, 1.0]
EPOPCHS =      [50] #, 100, 100,  100, 60]
TRAINED = {}

scale_factor = 1.0 #10 #output is zero all the time, a large scale factor will increase the penalty of that output

if __name__ == '__main__':

    for ar, n_epochs in zip(ANGLE_RATIOS, EPOPCHS):
        geometry = Geometry(ar, 200, 100, reco_shape=(256, 256))
        geom2 = Geometry(1.0, 200, 100)
        train_sinos, train_y, test_sinos, test_y = setup(geometry, num_to_generate=0, train_ratio=0.97, use_realistic=True, data_path="data/kits_phantoms_256.pt")
        #model = FBPNet(geometry, use_smooth_filters=False, n_fbps=2)
        # model = DoubleNet(geometry, use_smooth_filters=True)
        # model = CrazyKernels(geometry, angle_batch_size=4)
        # model = GeneralFBP(geometry)
        model = ChainedModels([RamLak(geometry), GeneralFBP(geom2)])

        #dataset = list(zip(train_sinos.to(DEVICE, non_blocking=True), train_y.to(DEVICE, non_blocking=True)))
        dataset = TensorDataset(train_sinos.cpu()*scale_factor, train_y.cpu()*scale_factor)

        multi_threaded_training(model, dataset, n_epochs=n_epochs, batch_size=20, lr=0.001, use_reg=False, num_threads=10)

        TRAINED[ar] = model
        model.visualize_output(test_sinos, test_y, output_location="show")
        torch.save(model.state_dict(), "testing.pt")
        print("Done")
    

