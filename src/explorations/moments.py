import torch
import numpy as np
from sklearn.linear_model import Ridge

from utils.geometry import ParallelGeometry, extend_geometry, DEVICE
from utils.inverse_moment_transform import get_Xn, get_Un
from utils.moments import SinoMoments


g = ParallelGeometry(0.5, 450, 300)
ext_g = extend_geometry(g)
smp = SinoMoments(ext_g, 12)


for ni in range(12):
    Xn = get_Xn(ext_g.tangles, ni)

    p_mom = smp.project_moment(Xn.T, ni)


    d, l = smp.on_basis[ni].shape
    dists = []
    for bindex in range(d):
        bi = smp.on_basis[ni][bindex]
        mdl = Ridge(0.0)
        mdl.fit(Xn, bi)
        bout = torch.from_numpy(mdl.predict(Xn)).to(DEVICE, dtype=torch.float)
        dists.append(torch.mean((bout-bi)**2).item())

    print(ni, "MSE projection Xn onto oni:", torch.mean((Xn.T-p_mom)**2).item(), "reverse:", np.mean(dists))


ts = torch.from_numpy(g.translations).to(DEVICE, dtype=torch.float)
ss = ts / g.rho
W = torch.sqrt(1 - ss**2)
N = np.pi*g.rho / 2

chebs = [get_Un(ss, ni) for ni in range(50)]

prods = torch.zeros(50, 50)

for i in range(50):
    for j in range(50):
        prods[i,j] = torch.sum(W*chebs[i]*chebs[j])*g.dt
        
    print()
print(torch.eye(5))

print(torch.mean((torch.eye(50)-prods/N)**2))
