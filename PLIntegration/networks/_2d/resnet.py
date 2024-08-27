#!usr/bin/env python
# -*- coding:utf-8 _*-

import math

import torch
from torch import nn


class SEBlock(nn.Module):
    """
    in_channels: ...
    reduction: default 16
    """

    def __init__(self, in_channels, reduction=16, **kwargs):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.PReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=(1, 1), downsample=False, use_se=False, **kwargs):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1)
        self.bn0 = nn.BatchNorm2d(out_channels)
        self.prelu0 = nn.PReLU()
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.prelu1 = nn.PReLU()
        if downsample:
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride)
            self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv0(x)
        out = self.bn0(out)
        out = self.prelu0(out)
        out = self.conv1(out)
        out = self.bn1(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample:
            residual = self.conv2(x)
            residual = self.bn2(residual)

        out += residual
        out = self.prelu1(out)

        return out


class BasicBlockV2(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=(1, 1), downsample=False, use_se=False, **kwargs):
        super().__init__()

        self.bn0 = nn.BatchNorm2d(in_channels)
        self.prelu0 = nn.PReLU()
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.prelu1 = nn.PReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        if downsample:
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(out_channels)

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.prelu0(out)
        out = self.conv0(out)

        out = self.bn1(out)
        out = self.prelu1(out)
        out = self.conv1(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample:
            residual = self.conv2(residual)

        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=False, use_se=False, **kwargs):
        """ Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
        """
        super().__init__()

        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.bn0 = nn.BatchNorm2d(out_channels)
        self.prelu0 = nn.PReLU()
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=(1, 1), stride=(1, 1),
                               padding=0)
        self.bn2 = nn.BatchNorm2d(self.expansion * out_channels)
        self.prelu2 = nn.PReLU()
        if downsample:
            self.conv3 = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=(1, 1), stride=stride)
            self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(self.expansion * out_channels)

    def forward(self, x):
        residual = x
        out = self.conv0(x)
        out = self.prelu0(self.bn0(out))
        out = self.conv1(out)
        out = self.prelu1(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample:
            residual = self.conv3(x)
            residual = self.bn3(residual)
        out = out + residual
        out = self.prelu2(out)
        return out


class BottleneckV2(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=False, use_se=False, **kwargs):
        """ Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
        """
        super().__init__()
        self.bn0 = nn.BatchNorm2d(in_channels)
        self.prelu0 = nn.PReLU()
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.prelu1 = nn.PReLU()
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.prelu2 = nn.PReLU()
        self.conv2 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=(1, 1), stride=(1, 1),
                               padding=0)
        if downsample:
            self.conv3 = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=(1, 1), stride=stride)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(self.expansion * out_channels)

    def forward(self, x):
        residual = x

        out = self.prelu0(self.bn0(x))
        out = self.conv0(out)
        out = self.prelu1(self.bn1(out))
        out = self.conv1(out)
        out = self.prelu2(self.bn2(out))
        out = self.conv2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample:
            residual = self.conv3(x)
        out = out + residual
        return out


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, start_block=False, end_block=False,
                 exclude_bn0=False, **kwargs):
        super().__init__()
        if not start_block and not exclude_bn0:
            self.bn0 = nn.BatchNorm2d(in_channels)
        if self.start_block:
            pass
        elif self.exclude_bn0:
            self.prelu0 = nn.PReLU()
        else:
            self.prelu1 = nn.PReLU()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.prelu2 = nn.PReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)

        if start_block:
            self.bn2 = nn.BatchNorm2d(out_channels)

        if end_block:
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.prelu3 = nn.PReLU()

        self.downsample = downsample
        self.stride = stride

        self.start_block = start_block
        self.end_block = end_block
        self.exclude_bn0 = exclude_bn0

    def forward(self, x):
        identity = x

        if self.start_block:
            out = self.conv1(x)
        elif self.exclude_bn0:
            out = self.prelu0(x)
            out = self.conv1(out)
        else:
            out = self.bn0(x)
            out = self.prelu1(out)
            out = self.conv1(out)

        out = self.bn1(out)
        out = self.prelu2(out)

        out = self.conv2(out)

        if self.start_block:
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if self.end_block:
            out = self.bn2(out)
            out = self.prelu3(out)

        return out


class IBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 start_block=False, end_block=False, exclude_bn0=False, **kwargs):
        super().__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        if not start_block and not exclude_bn0:
            self.bn0 = nn.BatchNorm2d(in_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)

        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.PReLU()
        self.downsample = downsample
        self.stride = stride

        self.start_block = start_block
        self.end_block = end_block
        self.exclude_bn0 = exclude_bn0

    def forward(self, x):
        identity = x

        if self.start_block:
            out = self.conv1(x)
        elif self.exclude_bn0:
            out = self.relu(x)
            out = self.conv1(out)
        else:
            out = self.bn0(x)
            out = self.relu(out)
            out = self.conv1(out)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.start_block:
            out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if self.end_block:
            out = self.bn3(out)
            out = self.relu(out)

        return out


class XBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, cardinality=32, base_width=4, downsample=False,
                 use_se=False,
                 **kwargs):
        """ Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups. default 32.
            base_width: base number of channels in each group. default 4.
        """
        super().__init__()
        D = cardinality * int(out_channels * (base_width / 64))
        self.conv0 = nn.Conv2d(in_channels, D, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.bn0 = nn.BatchNorm2d(D)
        self.prelu0 = nn.PReLU()
        self.conv1 = nn.Conv2d(D, D, kernel_size=(3, 3), stride=stride, padding=1, groups=cardinality)
        self.bn1 = nn.BatchNorm2d(D)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(D, self.expansion * out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.bn2 = nn.BatchNorm2d(self.expansion * out_channels)
        self.prelu2 = nn.PReLU()
        if downsample:
            self.conv3 = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=(1, 1), stride=stride)
            self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(self.expansion * out_channels)

    def forward(self, x):
        residual = x
        out = self.conv0(x)
        out = self.prelu0(self.bn0(out))
        out = self.conv1(out)
        out = self.prelu1(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample:
            residual = self.conv3(x)
            residual = self.bn3(residual)
        out = out + residual
        out = self.prelu2(out)
        return out


class XBottleneckV2(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, cardinality=32, base_width=4,
                 downsample=False, use_se=False, **kwargs):
        """ Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups. default 32.
            base_width: base number of channels in each group. default 4.
        """
        super().__init__()
        D = cardinality * int(out_channels * (base_width / 64))
        self.bn0 = nn.BatchNorm2d(D)
        self.prelu0 = nn.PReLU()
        self.conv0 = nn.Conv2d(in_channels, D, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.bn1 = nn.BatchNorm2d(D)
        self.prelu1 = nn.PReLU()
        self.conv1 = nn.Conv2d(D, D, kernel_size=(3, 3), stride=stride, padding=1, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(D)
        self.prelu2 = nn.PReLU()
        self.conv2 = nn.Conv2d(D, self.expansion * out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)
        if downsample:
            self.conv3 = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=(1, 1), stride=stride)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(self.expansion * out_channels)

    def forward(self, x):
        residual = x

        out = self.prelu0(self.bn0(x))
        out = self.conv0(out)
        out = self.prelu1(self.bn1(out))
        out = self.conv1(out)
        out = self.prelu2(self.bn2(out))
        out = self.conv2(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample:
            residual = self.conv3(x)
        out = out + residual
        return out


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=(1, 1), base_width=26, scale=4, stype='normal',
                 downsample=False, use_se=False, **kwargs):
        """ Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            base_width: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super().__init__()

        width = int(math.floor(out_channels * (base_width / 64.0)))
        self.conv1 = nn.Conv2d(in_channels, width * scale, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=(3, 3), stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, out_channels * self.expansion, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.stype = stype
        self.scale = scale
        self.width = width

        self.downsample = downsample
        if downsample:
            self.conv4 = nn.Conv2d(in_channels, out_channels * self.expansion,
                                   kernel_size=(1, 1), stride=stride, bias=False)
            self.bn4 = nn.BatchNorm2d(out_channels * self.expansion)
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(out_channels * self.expansion)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample:
            residual = self.conv4(x)
            residual = self.bn4(residual)

        out += residual
        out = self.relu(out)

        return out


class Bottle2neckV1b(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=(1, 1), base_width=26, scale=4, stype='normal',
                 downsample=False, use_se=False, **kwargs):
        """ Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            base_width: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super().__init__()

        width = int(math.floor(out_channels * (base_width / 64.0)))
        self.conv1 = nn.Conv2d(in_channels, width * scale, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        prelus = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=(3, 3), stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
            prelus.append(nn.PReLU())
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.prelus = nn.ModuleList(prelus)

        self.conv3 = nn.Conv2d(width * scale, out_channels * self.expansion, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.stype = stype
        self.scale = scale
        self.width = width

        self.downsample = downsample
        if downsample:
            self.avgpool = nn.AvgPool2d(kernel_size=stride, stride=stride,
                                        ceil_mode=True, count_include_pad=False)
            self.conv4 = nn.Conv2d(in_channels, out_channels * self.expansion,
                                   kernel_size=(1, 1), stride=1, bias=False)
            self.bn4 = nn.BatchNorm2d(out_channels * self.expansion)
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(out_channels * self.expansion)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.prelus[i](self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample:
            residual = self.avgpool(x)
            residual = self.conv4(residual)
            residual = self.bn4(residual)

        out += residual
        out = self.prelu2(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, input_channels, iresnet=False, **kwargs):
        super(ResNet, self).__init__()
        if kwargs.get("base_width", None) is None:
            kwargs["base_width"] = 4 if block in (XBottleneck, XBottleneckV2) else 26
        self.inplanes = 64
        if block == Bottle2neckV1b:
            self.conv1 = nn.Sequential(nn.Conv2d(input_channels, 32, 3, 2, 1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(32, 32, 3, 1, 1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(32, 64, 3, 1, 1, bias=False))
        else:
            self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.iresnet = iresnet
        make_layer = self._make_layer
        if iresnet:
            make_layer = self._make_ilayer
            self.layer1 = make_layer(block, 64, layers[0], stride=2, **kwargs)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = make_layer(block, 64, layers[0], **kwargs)
        self.layer2 = make_layer(block, 128, layers[1], stride=2, **kwargs)
        self.layer3 = make_layer(block, 256, layers[2], stride=2, **kwargs)
        self.layer4 = make_layer(block, 512, layers[3], stride=2, **kwargs)
        self.bn2 = nn.BatchNorm2d(2048)

    def _make_layer(self, block, planes, blocks, stride=1, cardinality=32, base_width=4, scale=4, use_se=False,
                    **kwargs):
        downsample = False
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = True
        layers = []
        layers.append(block(**{"in_channels": self.inplanes,
                               "out_channels": planes,
                               "stride": stride,
                               "downsample": downsample,
                               "cardinality": cardinality,
                               "base_width": base_width,
                               "scale": scale,
                               "use_se": use_se,
                               "stype": "stage"}))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(**{"in_channels": self.inplanes,
                                   "out_channels": planes,
                                   "cardinality": cardinality,
                                   "base_width": base_width,
                                   "scale": scale,
                                   "use_se": use_se}))
        return nn.Sequential(*layers)

    def _make_ilayer(self, block, planes, blocks, stride=1, **kwargs):
        downsample = None
        if stride != 1 and self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        elif self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        elif stride != 1:
            downsample = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, start_block=True))
        self.inplanes = planes * block.expansion
        exclude_bn0 = True
        for _ in range(1, (blocks - 1)):
            layers.append(block(self.inplanes, planes, exclude_bn0=exclude_bn0))
            exclude_bn0 = False

        layers.append(block(self.inplanes, planes, end_block=True, exclude_bn0=exclude_bn0))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        if not self.iresnet:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        return x


def resnet50(args):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **args)
    return model


def resnet50_v2(**kwargs):
    model = ResNet(BottleneckV2, [3, 4, 6, 3], **kwargs)
    return model


def resnext50(**kwargs):
    model = ResNet(XBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def res2net50(args):
    model = ResNet(Bottle2neck, [3, 4, 6, 3], **args)
    return model


def res2net50_v1b(**kwargs):
    model = ResNet(Bottle2neckV1b, [3, 4, 6, 3], **kwargs)
    return model


def iresnet50(**kwargs):
    model = ResNet(IBottleneck, [3, 4, 6, 3], iresnet=True, **kwargs)
    return model
