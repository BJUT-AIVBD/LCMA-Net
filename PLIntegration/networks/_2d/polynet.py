#!usr/bin/env python
# -*- coding:utf-8 _*-

import torch
from torch import nn
import torch.nn.functional as F
from typing import List


# region    # <--- module_start ---> #

class PolyBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1),
                 downsample=False):
        super(PolyBlock, self).__init__()
        # branch resnet
        self.bn0 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # branch polynet
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        b1_out = self.bn0(x)
        b1_out = self.conv1(b1_out)
        b1_out = self.bn1(b1_out)
        b1_out = self.prelu(b1_out)
        b1_out = self.conv2(b1_out)
        b1_out = self.bn2(b1_out)
        b2_out = x
        if self.downsample:
            b2_out = self.conv3(x)
            b2_out = self.bn3(b2_out)
            out = b1_out + b2_out
        else:
            out = b1_out + x
        b2_out = b1_out * b2_out
        b2_out = self.tanh(b2_out)
        b2_out = self.conv4(b2_out)
        b2_out = self.bn4(b2_out)
        out = b2_out + out

        return out


class PolyNet(nn.Module):
    def __init__(self, input_channels: int, filts: List[int], layers: List[int], dropout: float = 0, **kwargs):
        super(PolyNet, self).__init__()
        self.in_channels = filts[0]
        self.conv1 = nn.Conv2d(input_channels, filts[0],
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(filts[0])
        self.prelu = nn.PReLU()
        self.layer1 = self._make_layer(PolyBlock, filts[0], layers[0], stride=2)
        self.layer2 = self._make_layer(PolyBlock, filts[1], layers[1], stride=2)
        self.layer3 = self._make_layer(PolyBlock, filts[2], layers[2], stride=2)
        self.layer4 = self._make_layer(PolyBlock, filts[3], layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(filts[3])
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = self.dropout(x)
        return x

    def _make_layer(self, block, out_channels, blocks, stride=(1, 1)):
        layers = [block(self.in_channels, out_channels, stride, downsample=True)]
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(self.in_channels, self.in_channels))
        return nn.Sequential(*layers)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

# endregion # <--- module_start ---> #
