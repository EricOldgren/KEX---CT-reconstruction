import torch
from torch.utils.data import DataLoader, TensorDataset
import odl.contrib.torch as odl_torch

import os

from utils.geometry import ParallelGeometry, setup,  DEVICE, extend_geometry
from utils.threaded_training import multi_threaded_training
from models.fbpnet import FBPNet
from models.modelbase import ChainedModels
from utils.fno_1d import FNO1d
from models.fouriernet import FNO_BP, GeneralizedFNO_BP as GFNO_BP
from models.fbps import FBP, GeneralizedFBP as GFBP
from models.analyticmodels import RamLak, ramlak_filter
from models.expnet import FNOExtrapolatingBP
from utils.moments import SinoMoments

ANGLE_RATIOS = [0.5]#, 0.85, 0.9, 0.95, 1.0]
EPOPCHS =      [40] #, 100, 100,  100, 60]
TRAINED = {}
MODEL_STORE_LOCATION = os.path.join("data", "gfno_bp1")

if __name__ == '__main__':

    for ar, n_epochs in zip(ANGLE_RATIOS, EPOPCHS):
        geometry = ParallelGeometry(ar, 150, 75, reco_shape=(256, 256))
        ext_geom = extend_geometry(geometry)
        ext_ray_layer = odl_torch.OperatorModule(ext_geom.ray)
        
        train_sinos, train_y, val_sinos, val_y = setup(geometry, num_to_generate=0, train_ratio=0.9, use_realistic=True, data_path="data/kits_phantoms_256.pt")
        full_train_sinos, full_val_sinos = ext_ray_layer(train_y), ext_ray_layer(val_y)
        model = FNOExtrapolatingBP(geometry, exp_fno_layers=[20,20])
        
        dataset = TensorDataset(train_sinos, train_y, full_train_sinos)

        multi_threaded_training(model, dataset, (val_sinos, val_y, full_val_sinos), n_epochs=n_epochs, batch_size=15, lr=0.0003, num_threads=1, exp_model=True)

        TRAINED[ar] = model
        model.visualize_output(val_sinos, val_y, output_location="show")
        fn = "model.pt"
        inp = input(f"Enter path to store model in (enter for {os.path.join(MODEL_STORE_LOCATION, fn)})")
        if inp != "":
            fp = inp
        else:
            fp = os.path.join(os.path.dirname(MODEL_STORE_LOCATION), fn)
        torch.save(model.state_dict(), fp)
        print(f"Model saved to {fp}")
    

