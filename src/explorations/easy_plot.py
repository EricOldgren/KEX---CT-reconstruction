import torch
import matplotlib.pyplot as plt

from typing import Union, Tuple

from utils.tools import pacth_split_image_batch, merge_patches
from src.geometries.data import get_htc2022_train_phantoms

phantoms = get_htc2022_train_phantoms()



step = 128

patches = pacth_split_image_batch(phantoms, step)
merged = merge_patches(patches, phantoms.shape[-2:], step)

phantoms[:, ::step, :] = -1.0
phantoms[:, :, ::step] = -1.0

plt.figure()
plt.imshow(phantoms.cpu()[0])
plt.colorbar()
plt.figure()
plt.subplot(221)
plt.imshow(patches.cpu()[0,0].reshape(step, step))
plt.subplot(222)
plt.imshow(patches.cpu()[0,1].reshape(step, step))
plt.subplot(223)
plt.imshow(patches.cpu()[0,2].reshape(step, step))
plt.subplot(224)
plt.imshow(patches.cpu()[0,3].reshape(step, step))
plt.figure()
plt.imshow(merged[0].cpu())



for i in plt.get_fignums():
    fig = plt.figure(i)
    title = fig._suptitle.get_text() if fig._suptitle is not None else f"fig{i}"
    plt.savefig(f"{title}.png")