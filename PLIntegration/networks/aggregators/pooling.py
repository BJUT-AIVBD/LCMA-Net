#!usr/bin/env python
# -*- coding:utf-8 _*-
from torch import nn


class AdaAvgPool(nn.Module):
    def __init__(self, in_features: int, embed_size: int, pool_dims=2, bias: bool = True, dropout: float = 0,
                 output_size=(1, 1),
                 linear=False,
                 **kwargs):
        super().__init__()
        if pool_dims == 2:
            self.avgpool = nn.AdaptiveAvgPool2d(output_size)
        elif pool_dims == 3:
            self.avgpool = nn.AdaptiveAvgPool3d(output_size)
        self.linear = linear
        if linear:
            self.fc = nn.Linear(in_features, embed_size, bias)
            self.bn = nn.BatchNorm1d(embed_size)
        else:
            assert in_features == embed_size
        if dropout != 0:
            self.dropout = nn.Dropout(dropout)

    def forward(self, inpt):
        x = self.avgpool(inpt).view(inpt.shape[0], -1)
        if self.linear:
            x = self.fc(x)
            x = self.bn(x)
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        return x
