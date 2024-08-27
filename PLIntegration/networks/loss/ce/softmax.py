#!usr/bin/env python
# -*- coding:utf-8 _*-

import torch.optim
from torch import nn


class OriginSoftmax(nn.Module):
    def __init__(self, embed_size, num_classes, bias=True, **kwargs):
        super().__init__()
        self.fc = nn.Linear(embed_size, num_classes, bias)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, embedding, ground_truth):
        probs = self.fc(embedding)
        loss = self.ce(probs, ground_truth)
        return loss
