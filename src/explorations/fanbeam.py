import torch
import odl
import numpy as np
import odl.contrib.torch as odl_torch
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

phantoms = torch.stack(torch.load("data/HTC2022/HTCTrainingPhantoms.pt"))[2].to(DEVICE)
data = torch.stack(torch.load("data/HTC2022/HTCTrainingData.pt"))[2].to(DEVICE)

reco_space = odl.uniform_discr(
    min_pt=[-40, -40], max_pt=[40, 40], shape=[512, 512],
    dtype='float32')
angle_partition = odl.uniform_partition(0, 2 * np.pi, 721)
detector_partition = odl.uniform_partition(-56, 56, 560)
geometry = odl.tomo.FanBeamGeometry(
    angle_partition, detector_partition, src_radius=410.66, det_radius=143.08)


# --- Create Filtered Back-projection (FBP) operator --- #

ray_trafo = odl.tomo.RayTransform(reco_space, geometry)
fbp = odl.tomo.fbp_op(ray_trafo, filter_type='Hann', frequency_scaling=0.8)

Ray = odl_torch.OperatorModule(ray_trafo)
FBP = odl_torch.OperatorModule(fbp)

sino = Ray(phantoms[None])[0]
recon = FBP(sino[None])[0]

plt.imshow(sino.cpu())
plt.title("simulated")
plt.colorbar()
plt.show()
plt.imshow(data.cpu())
plt.title("gt")
plt.colorbar()
plt.show()
plt.imshow(phantoms.cpu())
plt.colorbar()
plt.title("gt")
plt.show()
plt.imshow(recon.cpu())
plt.colorbar()
plt.title("recon")
plt.show()




