#!usr/bin/env python
# -*- coding:utf-8 _*-
import torch.nn
from torch import nn


class Flatten(nn.Module):
    def __init__(self, in_features: int, embed_size: int,
                 start_dim: int = 1, end_dim: int = -1, bias: bool = True, dropout: float = 0, **kwargs):
        super().__init__()
        self.flatten = nn.Flatten(start_dim, end_dim)
        self.fc = nn.Linear(in_features, embed_size, bias)
        self.bn = nn.BatchNorm1d(embed_size)
        if dropout != 0:
            self.dropout = nn.Dropout(dropout)

    def forward(self, inpt):
        x = self.flatten(inpt)
        x = self.fc(x)
        x = self.bn(x)
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        return x
