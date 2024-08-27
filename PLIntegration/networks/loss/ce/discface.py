#!usr/bin/env python
# -*- coding:utf-8 _*-

import math

import torch
import torch.nn.functional as F
from torch import nn


class SoftmaxDiscFace(nn.Module):
    def __init__(self, embed_size, num_classes, lmbd=0.5, **kwargs):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_size))
        self.xi_bias = nn.Parameter(torch.FloatTensor(1, embed_size))
        self.lmbd = lmbd

        nn.init.zeros_(self.xi_bias)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding: torch.Tensor, ground_truth):
        norm_embd = F.normalize(embedding)
        norm_weight = F.normalize(self.weight)
        cos_theta = F.linear(embedding, self.weight).clamp(-1 + 1e-7, 1 - 1e-7)
        epsilon = norm_embd - norm_weight[ground_truth]
        norm_bias = self.xi_bias.norm().clamp(0, 0.05)
        disc_loss = (epsilon - F.normalize(self.xi_bias) * norm_bias).norm().mean()
        ce_loss = self.ce(cos_theta, ground_truth)
        return ce_loss + self.lmbd * disc_loss


class CosDiscFace(nn.Module):
    def __init__(self, embed_size, num_classes, scale=64, margin=0.35, lmbd=0.5, **kwargs):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.ce = nn.CrossEntropyLoss()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_size))
        self.xi_bias = nn.Parameter(torch.FloatTensor(1, embed_size))
        self.lmbd = lmbd

        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.xi_bias)

    def forward(self, embedding: torch.Tensor, ground_truth):
        norm_embd = F.normalize(embedding)
        norm_weight = F.normalize(self.weight)[ground_truth]
        cos_theta = F.linear(norm_embd, norm_weight).clamp(-1 + 1e-7, 1 - 1e-7)
        phi = torch.gather(cos_theta, 1, ground_truth.view(-1, 1)) - self.margin
        output = torch.scatter(cos_theta, 1, ground_truth.view(-1, 1).long(), phi)
        output *= self.scale
        ce_loss = self.ce(output, ground_truth)

        epsilon = norm_embd - norm_weight[ground_truth]
        norm_bias = self.xi_bias.norm().clamp(0, 0.05)
        disc_loss = (epsilon - F.normalize(self.xi_bias) * norm_bias).norm().mean()
        return ce_loss + self.lmbd * disc_loss


class ArcDiscFace(nn.Module):
    def __init__(self, embed_size, num_classes, scale=64, margin=0.5, easy_margin=False, lmbd=0.5, **kwargs):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.ce = nn.CrossEntropyLoss()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_size))
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        self.xi_bias = nn.Parameter(torch.FloatTensor(1, embed_size))
        self.lmbd = lmbd

        nn.init.zeros_(self.xi_bias)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding: torch.Tensor, ground_truth):
        """
        This Implementation is modified from forward1, which takes
        52.49644996365532 ms for every 100 times of input (50, 512) and output (50, 10000) on 2080 Ti.
        """
        norm_embd = F.normalize(embedding)
        norm_weight = F.normalize(self.weight)
        cos_theta = F.linear(norm_embd, norm_weight).clamp(-1 + 1e-7, 1 - 1e-7)
        pos = torch.gather(cos_theta, 1, ground_truth.view(-1, 1))
        sin_theta = torch.sqrt((1.0 - torch.pow(pos, 2)).clamp(0 + 1e-7, 1 - 1e-7))
        phi = pos * self.cos_m - sin_theta * self.sin_m
        if self.easy_margin:
            phi = torch.where(pos > 0, phi, pos)
        else:
            phi = torch.where(pos > self.th, phi, pos - self.mm)
        output = torch.scatter(cos_theta, 1, ground_truth.view(-1, 1).long(), phi)
        output *= self.scale
        ce_loss = self.ce(output, ground_truth)

        epsilon = norm_embd - norm_weight[ground_truth]
        norm_bias = self.xi_bias.norm().clamp(0, 0.05)
        disc_loss = (epsilon - F.normalize(self.xi_bias) * norm_bias).norm().mean()
        return ce_loss + self.lmbd * disc_loss


class RAMDiscSoftmax(nn.Module):
    def __init__(self, embed_size, num_classes, scale=30, margin=0.3, lmbd=0.5, **kwargs):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.ce = nn.CrossEntropyLoss()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_size))
        self.xi_bias = nn.Parameter(torch.FloatTensor(1, embed_size))
        self.lmbd = lmbd

        nn.init.zeros_(self.xi_bias)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding: torch.Tensor, ground_truth):
        norm_embd = F.normalize(embedding)
        norm_weight = F.normalize(self.weight)
        cos_theta = F.linear(norm_embd, norm_weight).clamp(-1 + 1e-7, 1 - 1e-7)
        cos_theta = cos_theta * self.scale
        phi = torch.gather(cos_theta, 1, ground_truth.view(-1, 1))
        one_hot = torch.scatter(torch.zeros_like(cos_theta), 1, ground_truth.view(-1, 1).long(), 1)
        # s_cos_theta = torch.scatter(cos_theta, 1, ground_truth.view(-1, 1).long(), phi)
        delta = cos_theta + self.scale * self.margin - phi
        delta = torch.masked_select(delta, one_hot != 1)
        delta = delta.reshape(cos_theta.shape[0], cos_theta.shape[1] - 1)
        delta = torch.where(delta > 0, delta, torch.zeros_like(delta))
        delta = torch.einsum("ij -> i", delta)
        ce_loss = torch.mean(torch.log(1 + delta))

        epsilon = norm_embd - norm_weight[ground_truth]
        norm_bias = self.xi_bias.norm().clamp(0, 0.05)
        disc_loss = (epsilon - F.normalize(self.xi_bias) * norm_bias).norm().mean()
        return ce_loss + self.lmbd * disc_loss
