import torch
from torch import nn
import torch.nn.functional as F

import math
import numpy as np

class Pointnet2d(nn.Module):
    def __init__(self,num_classes=3,input_channels=64):
        super().__init__()

