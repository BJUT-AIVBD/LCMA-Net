#!usr/bin/env python
# -*- coding:utf-8 _*-

import torch
from torch import nn


class SeqFace21(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv1a = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=0, bias=True)
        self.relu1a = nn.PReLU(num_parameters=32)
        self.conv1b = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=0,
                                bias=True)
        self.relu1b = nn.PReLU(num_parameters=64)
        self.pool1b = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1,
                                 bias=True)
        self.relu2_1 = nn.PReLU(num_parameters=64)
        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1,
                                 bias=True)
        self.relu2_2 = nn.PReLU(num_parameters=64)
        self.res2_2 = torch.add(64, 64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=0,
                               bias=True)
        self.relu2 = nn.PReLU(num_parameters=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1,
                                 bias=True)
        self.relu3_1 = nn.PReLU(num_parameters=128)
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1,
                                 bias=True)
        self.relu3_2 = nn.PReLU(num_parameters=128)
        self.res3_2 = torch.add(128, 128)
        self.conv3_3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1,
                                 bias=True)
        self.relu3_3 = nn.PReLU(num_parameters=128)
        self.conv3_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1,
                                 bias=True)
        self.relu3_4 = nn.PReLU(num_parameters=128)
        self.res3_4 = torch.add(128, 128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=0,
                               bias=True)
        self.relu3 = nn.PReLU(num_parameters=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1,
                                 bias=True)
        self.relu4_1 = nn.PReLU(num_parameters=256)
        self.conv4_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1,
                                 bias=True)
        self.relu4_2 = nn.PReLU(num_parameters=256)
        self.res4_2 = torch.add(256, 256)
        self.conv4_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1,
                                 bias=True)
        self.relu4_3 = nn.PReLU(num_parameters=256)
        self.conv4_4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1,
                                 bias=True)
        self.relu4_4 = nn.PReLU(num_parameters=256)
        self.res4_4 = torch.add(256, 256)
        self.conv4_5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1,
                                 bias=True)
        self.relu4_5 = nn.PReLU(num_parameters=256)
        self.conv4_6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1,
                                 bias=True)
        self.relu4_6 = nn.PReLU(num_parameters=256)
        self.res4_6 = torch.add(256, 256)
        self.conv4_7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1,
                                 bias=True)
        self.relu4_7 = nn.PReLU(num_parameters=256)
        self.conv4_8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1,
                                 bias=True)
        self.relu4_8 = nn.PReLU(num_parameters=256)
        self.res4_8 = torch.add(256, 256)
        self.conv4_9 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1,
                                 bias=True)
        self.relu4_9 = nn.PReLU(num_parameters=256)
        self.conv4_10 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1,
                                  bias=True)
        self.relu4_10 = nn.PReLU(num_parameters=256)
        self.res4_10 = torch.add(256, 256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=0,
                               bias=True)
        self.relu4 = nn.PReLU(num_parameters=512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1,
                                 bias=True)
        self.relu5_1 = nn.PReLU(num_parameters=512)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1,
                                 bias=True)
        self.relu5_2 = nn.PReLU(num_parameters=512)
        self.res5_2 = torch.add(512, 512)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1,
                                 bias=True)
        self.relu5_3 = nn.PReLU(num_parameters=512)
        self.conv5_4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1,
                                 bias=True)
        self.relu5_4 = nn.PReLU(num_parameters=512)
        self.res5_4 = torch.add(512, 512)
        self.conv5_5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1,
                                 bias=True)
        self.relu5_5 = nn.PReLU(num_parameters=512)
        self.conv5_6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=1,
                                 bias=True)
        self.relu5_6 = nn.PReLU(num_parameters=512)
        self.res5_6 = torch.add(512, 512)
        self.fc5 = nn.Linear(in_features=18432, out_features=512, bias=True)
        state_dict = torch.load(
            "/home/yjc/PythonProject/PLIntegration/PLIntegration/networks/_2d/third-party/SeqFace21.pt")
        self.load_state_dict(state_dict)

    def forward(self, inpt):
        x1 = self.relu1a(self.conv1a(inpt))
        x1 = self.relu1b(self.conv1b(x1))
        x1 = self.pool1b(x1)
        x2 = self.relu2_1(self.conv2_1(x1))
        x2 = self.relu2_2(self.conv2_2(x2))
        x2 = x1 + x2
        x2 = self.relu2(self.conv2(x2))
        x2 = self.pool2(x2)
        x3 = self.relu3_1(self.conv3_1(x2))
        x3 = self.relu3_2(self.conv3_2(x3))
        x2 = x2 + x3
        x3 = self.relu3_3(self.conv3_3(x2))
        x3 = self.relu3_4(self.conv3_4(x3))
        x3 = x2 + x3
        x3 = self.relu3(self.conv3(x3))
        x3 = self.pool3(x3)
        x4 = self.relu4_1(self.conv4_1(x3))
        x4 = self.relu4_2(self.conv4_2(x4))
        x3 = x3 + x4
        x4 = self.relu4_3(self.conv4_3(x3))
        x4 = self.relu4_4(self.conv4_4(x4))
        x3 = x3 + x4
        x4 = self.relu4_5(self.conv4_5(x3))
        x4 = self.relu4_6(self.conv4_6(x4))
        x3 = x3 + x4
        x4 = self.relu4_7(self.conv4_7(x3))
        x4 = self.relu4_8(self.conv4_8(x4))
        x3 = x3 + x4
        x4 = self.relu4_9(self.conv4_9(x3))
        x4 = self.relu4_10(self.conv4_10(x4))
        x4 = x3 + x4
        x4 = self.relu4(self.conv4(x4))
        x4 = self.pool4(x4)
        x5 = self.relu5_1(self.conv5_1(x4))
        x5 = self.relu5_2(self.conv5_2(x5))
        x4 = x4 + x5
        x5 = self.relu5_3(self.conv5_3(x4))
        x5 = self.relu5_4(self.conv5_4(x5))
        x4 = x4 + x5
        x5 = self.relu5_5(self.conv5_5(x4))
        x5 = self.relu5_6(self.conv5_6(x5))
        x5 = x4 + x5
        embed = self.fc5(x5.view(x5.shape[0], -1))
        return embed


if __name__ == '__main__':
    model = SeqFace21()
