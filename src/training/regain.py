from pathlib import Path
from utils.tools import GIT_ROOT
import re
import shutil

save_gen = lambda ar: GIT_ROOT /f"data/models/highscores_fno2/{ar:2}"
source = GIT_ROOT/"data/models"
pattern = re.compile(r"fnobp2_ar(0.\d+)_val(\d+?(\.\d*)?(e\-\d+))")


high_scores = {}
ar_lvl_map = {
    181/720: 1, 161/720: 2, 141/720: 3, 121/720: 4, 101/720: 5, 81/720: 6, 61/720: 7
}
for nm in source.glob("*"):
    res = pattern.search(nm.name)
    if res is not None:
        ar, valloss, _, _ = res.groups()
        ar, valloss = float(ar), float(valloss)

        if ar not in high_scores or high_scores[ar][0] > valloss:
            high_scores[ar] = (valloss, nm)

for ar, (valloss, nm) in high_scores.items():
    shutil.copy(nm, save_gen(ar)/"fnobp2.pt")
    print("Copied", nm, "to", save_gen(ar))