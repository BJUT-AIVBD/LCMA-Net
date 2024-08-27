#!usr/bin/env python
# -*- coding:utf-8 _*-

import torch
from torch import nn

act_dict = {"relu": nn.ReLU,
            "gelu": nn.GELU,
            "no": nn.Identity}


class MLB(nn.Module):
    """
    Multi-modal Low Rank Bilinear Pooling
    """

    def __init__(self, in_features: list, mid_features: int, fc_size, dim=1, act="relu", dropout=0.5, resize=True,
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.in_features = in_features
        if resize:
            for i, size in enumerate(in_features):
                self.__setattr__("resize_fc{}".format(i + 1), nn.Linear(size, mid_features))
                self.__setattr__("resize_act{}".format(i + 1), act_dict[act]())
        else:
            for i, size in enumerate(in_features):
                self.__setattr__("resize_fc{}".format(i + 1), nn.Identity())
                self.__setattr__("resize_act{}".format(i + 1), nn.Identity())

        self.fc_size = fc_size
        for i, size in enumerate(fc_size):
            self.__setattr__("fc{}".format(i + 1), nn.Linear(mid_features if i == 0 else fc_size[i - 1], size))
            self.__setattr__("act{}".format(i + 1), act_dict[act]())
            self.__setattr__("dropout{}".format(i + 1), nn.Dropout(dropout))
        self.bn = nn.BatchNorm1d(fc_size[-1])

    def forward(self, inpt):
        for i, size in enumerate(self.in_features):
            inpt[i] = self.__getattr__("fc{}".format(i + 1))(inpt[i])
            inpt[i] = self.__getattr__("act{}".format(i + 1))(inpt[i])

        for i, _ in enumerate(self.fc_size):
            x = self.__getattr__("fc{}".format(i + 1))(x)
            x = self.__getattr__("act{}".format(i + 1))(x)
            x = self.__getattr__("dropout{}".format(i + 1))(x)
        embed = self.bn(x)
        return embed
