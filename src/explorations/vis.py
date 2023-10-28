import torch
import matplotlib.pyplot as plt
from pathlib import Path
from utils.tools import segment_imgs

result_folder = Path("data/htc_results")
ar_map = {
    0.25: 90,
    0.22: 80,
    0.2: 70,
    0.17: 60,
    0.14: 50,
    0.11: 40,
    0.085: 30
}

for child in result_folder.glob("*"):
    if child.is_dir():
        angle_span = ar_map[float((child.name))]
        loc = Path(f"data/vis")
        if not loc.exists():
            loc.mkdir(parents=True)
            
        pred = torch.load(child/"pred.pt", map_location="cpu")
        pred = segment_imgs(pred)
        gt = torch.load(child/"gt.pt", map_location="cpu")
        loaded = {"gt": gt, "pred":pred}
        
        for fn, data in loaded.items():
            for disp_ind in range(3):
                plt.imshow(data[disp_ind], cmap="gray")

                plt.gca().set_axis_off()
                plt.margins(0,0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.savefig(loc/f"{angle_span}_{fn}{disp_ind}.pdf", bbox_inches = 'tight',
                    pad_inches = 0)

# plt.show()
