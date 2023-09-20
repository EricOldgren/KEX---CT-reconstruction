import torch
import time
from pathlib import Path
import matplotlib
matplotlib.use("WebAgg")
import matplotlib.pyplot as plt

# phantoms = torch.load(r"C:\Users\salom\Documents\code\KTH\KEX---CT-reconstruction\data\synthetic_htc_data.pt", map_location="cpu")[:10]
phantoms = torch.load(r"C:\Users\salom\Documents\code\KTH\KEX---CT-reconstruction\data\synthetic_htc_data.pt", map_location="cpu")[:10]

for i in range(10):
    plt.figure()
    plt.imshow(phantoms[i])

plt.show()