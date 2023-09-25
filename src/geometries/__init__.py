from .geometry_base import *
from .parallel_geometry.parallel_geometry import *
from .fanbeam_geometry import *
from typing import List

AVAILABLE_FBP_GEOMETRIES: List[FBPGeometryBase]  = [ParallelGeometry, FlatFanBeamGeometry]


scale_factor = 1.0
htc_th = 0.02
htc_sino_var = 2e-2
htc_mean_attenuation = 0.03
HTC2022_GEOMETRY = FlatFanBeamGeometry(720, 560, 410.66*scale_factor, 543.74*scale_factor, 112.0*scale_factor, [-38*scale_factor,38*scale_factor, -38*scale_factor,38*scale_factor], [512, 512])