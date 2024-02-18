import matplotlib.pyplot as plt
from PIL import Image

from geometries.data import get_synthetic_htc_phantoms, get_htc_testdata

sinos, _, _, phantoms = get_htc_testdata(1)

plt.imshow(phantoms[0], cmap="gray")
plt.figure()
plt.imshow(sinos[0, :360].T, cmap="gray")

plt.gca().set_axis_off()
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.savefig(f"sin.png", bbox_inches = 'tight',
    pad_inches = 0)

plt.show()
