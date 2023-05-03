import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from utils.geometry import Geometry, setup,  DEVICE
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

scale_factor = 1.0 #10 #output is zero all the time, a large scale factor will increase the penalty of that output

if __name__ == '__main__':

    for ar, n_epochs in zip(ANGLE_RATIOS, EPOPCHS):
        geometry = Geometry(ar, 200, 100, reco_shape=(256, 256))
        ext_geom = Geometry(1.0, 320, 100, reco_shape=(256, 256))
        smp = SinoMoments(ext_geom)
        modes = torch.where(geometry.fourier_domain <= geometry.omega)[0].shape[0]
        geom2 = Geometry(1.0, 200, 100)
        train_sinos, train_y, test_sinos, test_y = setup(geometry, num_to_generate=0, train_ratio=0.9, use_realistic=True, data_path="data/kits_phantoms_256.pt")
        # model = FBPNet(geometry, n_fbps=2, use_bias=False, default_kernel=ramlak_filter(geometry))
        # model = DoubleNet(geometry, use_smooth_filters=True)
        # model = GeneralFBP(geometry)
        # model = FBP(geometry, kernel=ramlak_filter(geometry, dtype=torch.float64))
        # model = GFBP(geometry)
        # model = FNO_BP(geometry, 200, layer_widths=[], dtype=torch.float32)
        # fno = FNO1d(modes, 160, 320, layer_widths=[40,40], verbose=True, dtype=torch.float32)
        # model = GFNO_BP(geometry, fno, ext_geom, dtype=torch.float32)
        model = FNOExtrapolatingBP(geometry)
        # model = ChainedModels(FBP(geometry, initial_kernel=ramlak_filter(geometry, dtype=torch.complex64)), GFBP(geom2, dtype=torch.complex64))

        #dataset = list(zip(train_sinos.to(DEVICE, non_blocking=True), train_y.to(DEVICE, non_blocking=True)))
        dataset = TensorDataset(train_sinos.cpu()*scale_factor, train_y.cpu()*scale_factor)

        multi_threaded_training(model, dataset, n_epochs=n_epochs, batch_size=20, lr=0.0001, use_reg=False, num_threads=10)

        TRAINED[ar] = model
        model.visualize_output(test_sinos, test_y, output_location="show")
        fn = "model.pt"
        inp = input(f"Enter path to store model in (enter for {os.path.join(MODEL_STORE_LOCATION, fn)})")
        if inp != "":
            fp = inp
        else:
            fp = os.path.join(os.path.dirname(MODEL_STORE_LOCATION), fn)
        torch.save(model.state_dict(), fp)
        print(f"Model saved to {fp}")
    

