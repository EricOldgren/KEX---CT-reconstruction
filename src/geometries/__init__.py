from .geometry_base import *
from .parallel_geometry.parallel_geometry import *
from .fanbeam_geometry import *
from typing import List

AVAILABLE_FBP_GEOMETRIES: List[FBPGeometryBase]  = [ParallelGeometry, FlatFanBeamGeometry]

HTC2022_GEOMETRY = FlatFanBeamGeometry(720, 560, 410.66, 543.74, 112, [-43.5,43.5, -43.5, 43.5], [512, 512])