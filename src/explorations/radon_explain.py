import odl

import matplotlib.pyplot as plt
from utils.geometry import Geometry

from odl.phantom.transmission import shepp_logan
from odl.tomo.operators import RayTransform


g = Geometry(1.0, 450, 300)

sl = shepp_logan(g.reco_space, modified=True)

sino = g.ray(sl)

sl.show()

sino.show(force_show=True)
