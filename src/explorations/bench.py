import time
from geometries import HTC2022_GEOMETRY
from geometries.data import get_htc_traindata
from utils.polynomials import Chebyshev

geometry = HTC2022_GEOMETRY
sinos, phantoms = get_htc_traindata()
M, K = 50, 50
T = 30


start = time.time()
for _ in range(T):
    res = geometry.project_forward(phantoms)
forward_time = (time.time()-start) / T
print("Forward time:", forward_time)

start = time.time()
for _ in range(T):
    res = geometry.project_backward(sinos)
backward_time = (time.time()-start) / T
print("backward time:", backward_time)

start = time.time()
for _ in range(T):
    res = geometry.series_expand(sinos, Chebyshev, M, K)
expansion_time = (time.time()-start) / T
print("expansion time:", expansion_time)

coeffs = res
start = time.time()
for _ in range(T):
    res = geometry.synthesise_series(coeffs, Chebyshev)
synth_time = (time.time()-start) / T
print("synthesise time:", synth_time)
