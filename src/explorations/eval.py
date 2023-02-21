import torch
import os
import glob
import matplotlib.pyplot as plt
from src.utils.geometry import BasicModel, setup

folder = os.path.join("data", "kernels", "*")

kernels = glob.glob(folder)

loss_fn = lambda diff : torch.mean(diff*diff)

losses = []
ratios = []

plt.subplot(211)
for path in kernels:
    kernel = torch.load(path, map_location=torch.device("cpu"))
    ar = os.path.basename(path)[-6:-3]

    plt.plot(list(range(kernel.shape[0])), kernel.detach().cpu(), label=ar)

    (sinos, y, _, _), _ = setup(float(ar), num_samples=10, train_ratio=1.0)
    model = BasicModel(float(ar), 100, 300, kernel)
    out = model(sinos)
    losses.append(loss_fn(out - y).item())
    ratios.append(float(ar))
plt.legend()

plt.subplot(212)
plt.plot(ratios, losses, label="loss")
plt.ylabel("MSE loss")
plt.xlabel("Angle ratio")

plt.legend()
plt.show()