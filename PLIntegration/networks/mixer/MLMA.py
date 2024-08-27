#!usr/bin/env python
# -*- coding:utf-8 _*-

import torch
from torch import nn

act_dict = {"relu": nn.ReLU,
            "gelu": nn.GELU,
            "no": nn.Identity}


class MLMA(nn.Module):
    def __init__(self, in_features, fc_size, dim=1, **kwargs):
        super().__init__()
        self.dim = dim
        self.fc_size = fc_size
        self.conv1 = nn.Conv1d(in_features, 128, 1)
        self.conv2 = nn.Conv2d(128, 32, 1)

        # for i, size in enumerate(fc_size):
        #     self.__setattr__("fc{}".format(i + 1),
        #                      nn.Linear(in_features * in_modals if i == 0 else fc_size[i - 1], size))
        #     self.__setattr__("act{}".format(i + 1), act_dict[act]())
        #     self.__setattr__("dropout{}".format(i + 1), nn.Dropout(dropout))
        # self.bn = nn.BatchNorm1d(fc_size[-1])

    def forward(self, inpt):
        # inpt: BxDxM
        x = torch.relu(self.conv1(inpt.transpose(1, 2)))
        x = torch.relu(self.conv2(x))  # BxCxM
        gram = torch.softmax(torch.matmul(x.transpose(1, 2), x), -1)  # MxM
        embed = torch.matmul(inpt, gram).transpose(1, 2)  # BxDxM->BxMxD
        # x = torch.cat(x, dim=self.dim)
        # for i, _ in enumerate(self.fc_size):
        #     x = self.__getattr__("fc{}".format(i + 1))(x)
        #     x = self.__getattr__("act{}".format(i + 1))(x)
        #     x = self.__getattr__("dropout{}".format(i + 1))(x)
        # embed = self.bn(x)
        return embed
