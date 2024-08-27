#!usr/bin/env python
# -*- coding:utf-8 _*-

import torch
from torch import nn

act_dict = {"relu": nn.ReLU,
            "gelu": nn.GELU,
            "no": nn.Identity}


class simple_att(nn.Module):
    def __init__(self, input_dim, act="relu"):
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=input_dim)
        self.act1 = act_dict[act]()
        self.fc2 = nn.Linear(in_features=input_dim, out_features=input_dim)
        self.act2 = nn.Sigmoid()
    def forward(self, inpt):
        x = self.fc1(inpt)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x


att_dict = {"simple_att": simple_att}


class AdaptiveMultimodalFusion(nn.Module):
    """
    Adaptive multimodal fusion for facial action units recognition
    """

    def __init__(self, in_features, fc_size, dim=1, act="relu", dropout=0.5, **kwargs):
        super().__init__()
        self.dim = dim
        self.fc_size = fc_size
        for i, size in enumerate(fc_size):
            self.__setattr__("fc{}".format(i + 1), nn.Linear(in_features if i == 0 else fc_size[i - 1], size))
            self.__setattr__("act{}".format(i + 1), act_dict[act]())
            self.__setattr__("dropout{}".format(i + 1), nn.Dropout(dropout))
        self.bn = nn.BatchNorm1d(fc_size[-1])

    def forward(self, inpt):
        for i, _ in enumerate(self.fc_size):
            x = self.__getattr__("fc{}".format(i + 1))(x)
            x = self.__getattr__("act{}".format(i + 1))(x)
            x = self.__getattr__("dropout{}".format(i + 1))(x)
        embed = self.bn(x)
        return embed
