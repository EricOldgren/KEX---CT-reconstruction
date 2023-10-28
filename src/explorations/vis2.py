from pathlib import Path
import shutil

source = Path("data/HTC2022/TestData")
dest = Path("data/vis")
dest.mkdir(exist_ok=True)

lvls = {
    1:90,
    2:80,
    3:70,
    4:60,
    5:50,
    6:40,
    7:30
}

for lvl, angle_span in lvls.items():
    for disp_ind, cat in enumerate(["a","b", "c"]):
        shutil.copy(source/f"htc2022_0{lvl}{cat}_recon_fbp_seg_limited.png", dest/f"{angle_span}_fbp_limited{disp_ind}.png")