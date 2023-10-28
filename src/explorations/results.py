from pathlib import Path
import torch
from utils.tools import htc_score
import scipy

source = Path("data/HTC2022/TestData")
lvls = {
    1:90,
    2:80,
    3:70,
    4:60,
    5:50,
    6:40,
    7:30
}
scores = []
for lvl, angle_span in lvls.items():
    score = 0
    for disp_ind, cat in enumerate(["a","b", "c"]):
        recon = torch.from_numpy(scipy.io.loadmat((source/f"htc2022_0{lvl}{cat}_recon_fbp_seg_limited.mat"))["reconLimitedFbpSeg"]).to(bool)
        gt = torch.from_numpy(scipy.io.loadmat(source/f"htc2022_0{lvl}{cat}_recon_fbp_seg.mat")["reconFullFbpSeg"]).to(bool)
        score += htc_score(recon, gt).item()
    scores.append(score)

print(len(scores))
print(scores)