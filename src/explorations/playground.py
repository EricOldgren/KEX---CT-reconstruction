import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.geometry import Geometry, setup, BasicModel
from models.fbpnet import FBPNet
from models.fouriernet import CrazyKernels
import random

