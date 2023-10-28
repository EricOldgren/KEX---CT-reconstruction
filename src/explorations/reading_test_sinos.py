import torch
import scipy

from utils.tools import GIT_ROOT, DTYPE, DEVICE, htc_score, segment_imgs
from geometries import HTC2022_GEOMETRY, mark_cyclic
from geometries.data import htc_th


def get_htc_testdata(level: int):
    assert level >= 1 and level <= 7, "invalid level {level}"
    n_known_angles = [181, 161, 141, 121, 101, 81, 61]

    sinos = []
    to_rotate = []
    phantoms = []

    for c in ['a','b','c']:

        full_data = scipy.io.loadmat(GIT_ROOT/('data/HTC2022/TestData/htc2022_0' + str(level) + c + '_full.mat'))["CtDataFull"][0,0]
        full_sino = torch.tensor(full_data["sinogram"])
        dataLimited = scipy.io.loadmat(GIT_ROOT/('data/HTC2022/TestData/htc2022_0' + str(level) + c + '_limited.mat'))["CtDataLimited"][0,0]
        phantom = torch.tensor(scipy.io.loadmat(GIT_ROOT/('data/HTC2022/TestData/htc2022_0' + str(level) + c + '_recon_fbp_seg.mat'))['reconFullFbpSeg']).to(DEVICE, dtype=bool)
        la_sino = torch.tensor(dataLimited["sinogram"])
        assert la_sino.shape == (n_known_angles[level-1], HTC2022_GEOMETRY.projection_size)
        known_angles = torch.tensor(dataLimited["parameters"]["angles"][0,0])
        to_rotate.append(int(known_angles[0, 0] / 0.5))
        sinos.append(torch.concat([la_sino, torch.zeros((HTC2022_GEOMETRY.n_projections-n_known_angles[level-1], HTC2022_GEOMETRY.projection_size))]).to(DEVICE, dtype=DTYPE))
        phantoms.append(phantom)

        angles = torch.zeros(721, dtype=bool)
        angles = mark_cyclic(angles, to_rotate[-1], (to_rotate[-1]+n_known_angles[level-1])%721)

        assert (full_sino[angles] == la_sino).all()
        to_rotate[-1] += 180 #angle is offset by 90 degrees

        

    return torch.stack(sinos), to_rotate, torch.stack(phantoms)


import matplotlib.pyplot as plt

get_htc_testdata(1)

# for lvl in [1,2,3,4,5,6,7]:
#     sinos, to_rot, phantoms = get_htc_testdata(lvl)

# print("passed all tests!!!")

# disp_ind = 2
# plt.imshow(HTC2022_GEOMETRY.rotate_sinos(sinos, to_rot[disp_ind])[disp_ind])
# plt.figure()
# plt.imshow(phantoms[disp_ind].cpu())
# plt.show()