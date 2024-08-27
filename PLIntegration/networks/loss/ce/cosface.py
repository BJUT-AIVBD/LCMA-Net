#!usr/bin/env python
# -*- coding:utf-8 _*-
import torch
import torch.nn.functional as F
from torch import nn


class CosFace(nn.Module):
    def __init__(self, embed_size, num_classes, scale=30, margin=0.4, **kwargs):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.ce = nn.CrossEntropyLoss()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_size))

        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding: torch.Tensor, ground_truth):
        cos_theta = F.linear(F.normalize(embedding), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        phi = torch.gather(cos_theta, 1, ground_truth.view(-1, 1)) - self.margin
        output = torch.scatter(cos_theta, 1, ground_truth.view(-1, 1).long(), phi)
        output *= self.scale
        loss = self.ce(output, ground_truth)
        return loss
