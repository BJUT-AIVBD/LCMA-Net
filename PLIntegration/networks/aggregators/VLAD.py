#!usr/bin/env python
# -*- coding:utf-8 _*-
import torch
import torch.nn.functional as F
from torch import nn


class GhostVlad(nn.Module):
    def __init__(self, num_clusters=8, ghost=1, dim=128, alpha=100.0, normalize_input=True):
        super(GhostVlad, self).__init__()
        self.alpha = alpha
        self.num_clusters = num_clusters
        self.normalize_input = normalize_input
        self.ghost = ghost
        self.dim = dim
        self.conv = nn.Conv1d(dim, num_clusters + ghost, kernel_size=1, bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

    def forward(self, x):
        N, C = x.shape[:2]
        assert C == self.dim
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)
        soft_assign = self.conv(x).view(N, self.num_clusters + self.ghost, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        soft_assign = soft_assign[:, :self.num_clusters, :]
        x_flatten = x
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) \
                   - self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)  # (N, 9, 256, 26)

        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)  # (N, 9, 256)
        vlad = F.normalize(vlad, p=2, dim=2)
        vlad = vlad.view(x.size(0), -1)
        return vlad


class AttVlad(nn.Module):
    def __init__(self, num_clusters=8, dim=128, alpha=100.0, normalize_input=True):
        super(AttVlad, self).__init__()
        self.alpha = alpha
        self.num_clusters = num_clusters
        self.normalize_input = normalize_input
        self.dim = dim
        self.conv = nn.Conv1d(dim, num_clusters, kernel_size=1, bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self.att_fc = nn.Linear(dim, 1)

    def forward(self, x):
        N, C = x.shape[:2]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)
        soft_assign = self.conv(x)
        soft_assign = F.softmax(soft_assign, dim=1)
        soft_att = self.att_fc(self.centroids)

        x_flatten = x
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) \
                   - self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)  # (N, 9, 256, 26)

        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)  # (N, 9, 256)
        att_vlad = vlad * soft_att.unsqueeze(0)
        att_vlad = F.normalize(att_vlad, p=2, dim=2)
        # vlad = vlad.view(x.size(0), -1)
        return att_vlad


if __name__ == '__main__':
    input = torch.rand(2, 256, 16)
    model = AttVlad(dim=256)
    output = model(input)
    print(output.shape)
