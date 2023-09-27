import torch
import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt
from src.geometries.data import disc_phantom, GIT_ROOT, get_kits_train_phantoms, disc_phantom_rects
from geometries import htc_mean_attenuation
import numpy as np
import sys
from tqdm import tqdm

xy_minmax = [-38,38,-38,38]
disc_radius = 35
phantom_shape = (512, 512)

generated_synthetic_data = []
unit_size_ellips, unit_size_rects = 150, 100
settings_ellips = [
    (15, 0.9, 1.0, 1), #n_ellipses, min_ellips_ratio, max_ellips_ratio, n_phantom_units
    (20, 0.9, 1.0, 2),
    (10, 0.8, 1.0, 1),
    (20, 0.8, 1.0, 1),
    (25, 0.7, 0.8, 1),
    (50, 0.7, 0.8, 1),
    (30, 0.5, 0.6, 1),
    (50, 0.5, 0.6, 2),
]
settings_rects = [
    (10, 0.9, 1.0, 1), #n_rects, min_side_ratio, max_side_ratio, n_phantom_units
    (20, 0.9, 1.0, 2),
    (10, 0.8, 1.0, 1),
    (20, 0.8, 1.0, 1),
    (25, 0.7, 0.8, 1),
    (50, 0.7, 0.8, 1),
    (30, 0.5, 0.6, 1),
    (50, 0.5, 0.6, 2),
]

for n_ellipses, m, M, n_units in settings_ellips:
        N = n_units * unit_size_ellips
        for i in tqdm(range(N), desc="phantom generation"):
            mu = htc_mean_attenuation * (1 + np.random.randn()*0.1)
            phantom = disc_phantom(xy_minmax, disc_radius, phantom_shape, n_ellipses, m, M)*mu
            if i == 0:
                plt.figure()
                plt.imshow(phantom.cpu())
                plt.colorbar()
                plt.title(f"M,m,n_ph:{M},{m},{n_ellipses}")
            generated_synthetic_data.append(phantom)
        print("Ellips Data generated with settings:", n_ellipses, m, M, N)

for n_rects, m, M, n_units in settings_rects:
    N = n_units*unit_size_rects
    for i in tqdm(range(N), desc="phantom generation"):
        mu = htc_mean_attenuation * (1 + np.random.randn()*0.1)
        phantom = disc_phantom_rects(xy_minmax, disc_radius, phantom_shape, n_rects, m, M)*mu
        if i == 0:
            plt.figure()
            plt.imshow(phantom.cpu())
            plt.colorbar()
            plt.title(f"M,m,n_ph:{M},{m},{n_ellipses}")
        generated_synthetic_data.append(phantom)
    print("Rect Data generated with settings:", n_ellipses, m, M, N)


generated_synthetic_data = torch.stack(generated_synthetic_data)
save_path = GIT_ROOT / "data/synthetic_htc_bigbatch.pt"
torch.save(generated_synthetic_data, save_path)
print("Data saved to:", save_path)



plt.show()