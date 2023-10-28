import torch
import matplotlib.pyplot as plt

from utils.tools import GIT_ROOT, DEVICE, htc_score


preds = torch.load(GIT_ROOT/f"data/htc_results/0.14/pred.pt", map_location=DEVICE)
gts = torch.load(GIT_ROOT/f"data/htc_results/0.14/gt.pt", map_location=DEVICE)

print(htc_score(preds>0.02, gts).sum())

disp_ind = 0
plt.subplot(121)
plt.imshow((preds>0.02)[disp_ind].cpu())
plt.subplot(122)
plt.imshow(gts[disp_ind].cpu())

plt.figure()
plt.imshow(preds[disp_ind].cpu())

plt.show()