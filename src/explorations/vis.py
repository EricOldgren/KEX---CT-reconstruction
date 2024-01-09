import torch
import matplotlib.pyplot as plt
from pathlib import Path
from utils.tools import segment_imgs
from tqdm import tqdm

from geometries.data import htc_th

result_folder = Path("data/htc_results_samephantom")
save_folder = Path(f"data/vis_same_phantom")
ar_map = {
    0.25: 90,
    0.22: 80,
    0.2: 70,
    0.17: 60,
    0.14: 50,
    0.11: 40,
    0.085: 30
}
gt = torch.load(result_folder/"gt.pt", map_location="cpu")
gt = segment_imgs(gt)
for disp_ind in range(3):
    plt.imshow(gt[disp_ind], cmap="gray")

    plt.gca().set_axis_off()
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(save_folder/f"gt{disp_ind}.pdf", bbox_inches = 'tight',
        pad_inches = 0)
print("GT saved", end="\r")

for child in result_folder.glob("*"):
    if child.is_dir():
        angle_span = ar_map[float((child.name))]
        if not save_folder.exists():
            save_folder.mkdir(parents=True)

        fno = torch.load(child/"pred_model.pt", map_location="cpu")
        fbp = torch.load(child/"la_fbp.pt", map_location="cpu")
        mom_fbp = torch.load(child/"la_mom_fbb.pt", map_location="cpu")

        # fno, fbp, mom_fbp = fno>htc_th, fbp>htc_th, mom_fbp>htc_th
        # fno, fbp, mom_fbp = segment_imgs(fno), segment_imgs(fbp), segment_imgs(mom_fbp)
        loaded = {"fno":fno, "fbp": fbp, "mom_fbp": mom_fbp}
        
        for fn, data in loaded.items():
            for disp_ind in range(3):
                plt.imshow(data[disp_ind], cmap="gray")

                plt.gca().set_axis_off()
                plt.margins(0,0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.savefig(save_folder/f"{angle_span}_{fn}{disp_ind}.pdf", bbox_inches = 'tight',
                    pad_inches = 0)
                
        print(f"Images from {child} is saved.", end="\r")

# plt.show()
