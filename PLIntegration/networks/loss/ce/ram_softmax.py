#!usr/bin/env python
# -*- coding:utf-8 _*-
import torch
from torch import nn
import torch.nn.functional as F


class RealAMSoftmax(nn.Module):
    def __init__(self, embed_size, num_classes, scale=30, margin=0.3, **kwargs):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.ce = nn.CrossEntropyLoss()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_size))

        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding: torch.Tensor, ground_truth):
        cos_theta = F.linear(F.normalize(embedding), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        cos_theta = cos_theta * self.scale
        phi = torch.gather(cos_theta, 1, ground_truth.view(-1, 1))
        one_hot = torch.scatter(torch.zeros_like(cos_theta), 1, ground_truth.view(-1, 1).long(), 1)
        # s_cos_theta = torch.scatter(cos_theta, 1, ground_truth.view(-1, 1).long(), phi)
        delta = cos_theta + self.scale * self.margin - phi
        delta = torch.masked_select(delta, one_hot != 1)
        delta = delta.reshape(cos_theta.shape[0], cos_theta.shape[1] - 1)
        delta = torch.where(delta > 0, delta, torch.zeros_like(delta))
        delta = torch.einsum("ij -> i", delta)
        loss = torch.mean(torch.log(1 + delta))
        return loss


class MVRealAMSoftmax(nn.Module):
    def __init__(self, embed_size, num_classes, scale=30, margin=0.3, t=0.2, fixed=False, **kwargs):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.t = t
        self.fixed = fixed
        self.ce = nn.CrossEntropyLoss()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_size))

        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding: torch.Tensor, ground_truth):
        norm_embd = F.normalize(embedding)
        norm_weight = F.normalize(self.weight)
        cos_theta = F.linear(norm_embd, norm_weight).clamp(-1 + 1e-7, 1 - 1e-7)
        phi = torch.gather(cos_theta, 1, ground_truth.view(-1, 1))
        mask = cos_theta > phi - self.margin
        if self.fixed:
            cos_theta = torch.masked_scatter(cos_theta, mask, cos_theta[mask] + self.t)  # fixed
        else:
            cos_theta = torch.masked_scatter(cos_theta, mask, (self.t + 1.0) * cos_theta[mask] + self.t)  # adaptive
        one_hot = torch.scatter(torch.zeros_like(cos_theta), 1, ground_truth.view(-1, 1).long(), 1)

        delta = self.scale * (cos_theta + self.margin - phi)
        delta = torch.masked_select(delta, one_hot != 1)
        delta = delta.reshape(cos_theta.shape[0], cos_theta.shape[1] - 1)
        delta = torch.where(delta > 0, delta, torch.zeros_like(delta))
        delta = torch.einsum("ij -> i", delta)
        loss = torch.mean(torch.log(1 + delta))
        return loss


class FocalRealAMSoftmax(nn.Module):
    def __init__(self, embed_size, num_classes, scale=30, margin=0.3, gamma=2, **kwargs):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.gamma = gamma
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_size))

        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding: torch.Tensor, ground_truth):
        cos_theta = F.linear(F.normalize(embedding), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        cos_theta = cos_theta * self.scale
        phi = torch.gather(cos_theta, 1, ground_truth.view(-1, 1))
        one_hot = torch.scatter(torch.zeros_like(cos_theta), 1, ground_truth.view(-1, 1).long(), 1)
        delta = cos_theta + self.scale * self.margin - phi
        delta = torch.masked_select(delta, one_hot != 1)
        delta = delta.reshape(cos_theta.shape[0], cos_theta.shape[1] - 1)
        delta = torch.where(delta > 0, delta, torch.zeros_like(delta))
        delta = torch.einsum("ij -> i", delta)
        # p = 1 / (1 + delta)
        loss = torch.mean(torch.pow(1 - 1 / (1 + delta), self.gamma) * torch.log(1 + delta))
        return loss


class MVDiscRealAMSoftmax(nn.Module):
    def __init__(self, embed_size, num_classes, scale=30, margin=0.3, t=0.2, lmbd=0.5, fixed=False, **kwargs):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.t = t
        self.fixed = fixed
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_size))
        self.xi_bias = nn.Parameter(torch.FloatTensor(1, embed_size))
        self.lmbd = lmbd

        nn.init.zeros_(self.xi_bias)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding: torch.Tensor, ground_truth):
        norm_embd = F.normalize(embedding)
        norm_weight = F.normalize(self.weight)
        cos_theta = F.linear(norm_embd, norm_weight).clamp(-1 + 1e-7, 1 - 1e-7)
        phi = torch.gather(cos_theta, 1, ground_truth.view(-1, 1))
        mask = cos_theta > phi - self.margin
        if self.fixed:
            cos_theta = torch.masked_scatter(cos_theta, mask, cos_theta[mask] + self.t)  # fixed
        else:
            cos_theta = torch.masked_scatter(cos_theta, mask, (self.t + 1.0) * cos_theta[mask] + self.t)  # adaptive
        one_hot = torch.scatter(torch.zeros_like(cos_theta), 1, ground_truth.view(-1, 1).long(), 1)

        delta = self.scale * (cos_theta + self.margin - phi)
        delta = torch.masked_select(delta, one_hot != 1)
        delta = delta.reshape(cos_theta.shape[0], cos_theta.shape[1] - 1)
        delta = torch.where(delta > 0, delta, torch.zeros_like(delta))
        delta = torch.einsum("ij -> i", delta)
        loss = torch.mean(torch.log(1 + delta))

        epsilon = norm_embd - norm_weight[ground_truth]
        norm_bias = self.xi_bias.norm().clamp(0, 0.05)
        disc_loss = (epsilon - F.normalize(self.xi_bias) * norm_bias).norm().mean()
        return loss + self.lmbd * disc_loss
