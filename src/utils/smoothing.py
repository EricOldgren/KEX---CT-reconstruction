import torch
import torch.nn as nn
from .geometry import BasicModel


class SmoothedModel(BasicModel):

    def __init__(self) -> None:
        "FBP with kernel "