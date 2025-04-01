import os
import math

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
import csv


class FeedForwardNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def gen_feed_forward(n_features=10):
    return FeedForwardNet(input_dim=n_features, hidden_dim=n_features, output_dim=1)
