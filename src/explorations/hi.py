import matplotlib.pyplot as plt

from geometries.data import get_synthetic_htc_phantoms

phantoms = get_synthetic_htc_phantoms()

plt.subplot(121)
plt.imshow(phantoms[0] > 0)
plt.subplot(122)
plt.imshow(phantoms[0])

plt.show()