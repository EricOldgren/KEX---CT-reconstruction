import torch
import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt
from utils.data import disc_phantom, GIT_ROOT

xy_minmax = [-1,1,-1,1]
disc_radius = 0.8
phantom_shape = (512, 512)

generated_synthetic_data = []
settings = [
    (5, 0.9, 1.0, 50), #n_ellipses, min_ellips_ratio, max_ellips_ratio, n_phantoms
    (10, 0.9, 1.0, 50),
    (20, 0.9, 1.0, 50),
    (5, 0.8, 1.0, 50),
    (10, 0.8, 1.0, 50),
    (20, 0.8, 1.0, 50),
    (30, 0.5, 0.6, 50),
    (50, 0.5, 0.6, 50),
    (40, 0.4, 0.5, 50),
    (60, 0.4, 0.5, 50)
]

for n_ellipses, m, M, N in settings:
        for i in range(N):
            phantom = disc_phantom(xy_minmax, disc_radius, phantom_shape, n_ellipses, m, M)
            if i < 2:
                plt.figure()
                plt.imshow(phantom.cpu())
                plt.colorbar()
                plt.title(f"M,m,n_ph:{M},{m},{n_ellipses}")
            generated_synthetic_data.append(phantom)
        print("Data generated with settings:", n_ellipses, m, M, N)

plt.show()

generated_synthetic_data = torch.stack(generated_synthetic_data)
save_path = GIT_ROOT / "data/synthetic_htc_harder_data.pt"
torch.save(generated_synthetic_data, save_path)
print("Data saved to:", save_path)