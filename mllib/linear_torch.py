import os
import math

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
import csv


class LinearNet(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.net = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.net(x)


def gen_linear_torch(n_features=10):
    return LinearNet(input_dim=n_features, output_dim=1)
