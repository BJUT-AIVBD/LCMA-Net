#!usr/bin/env python
# -*- coding:utf-8 _*-
import math

import torch
import torch.nn.functional as F
from torch import nn


class MixSoftmax(nn.Module):
    def __init__(self, fc_type='MV-AM', loss_type="Softmax",
                 margin=0.35, scale=32, t=0.2, lampda=0.5,
                 gamma=2.0, save_rate=0.9,
                 embed_size=512, num_classes=72690,
                 easy_margin=False, **kwargs):
        super().__init__()
        # initial parameters
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_size))
        nn.init.xavier_uniform_(self.weight)
        if "Disc" in loss_type.split("+"):
            self.xi_bias = nn.Parameter(torch.FloatTensor(1, embed_size))
            nn.init.zeros_(self.xi_bias)

        # init fc attributes
        self.fc_type = fc_type
        self.scale = scale
        self.margin = margin
        self.t = t
        self.lampda = lampda
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        # duplication formula for SphereFace(A-Softmax)
        self.iter = 0
        self.base = 1000
        self.alpha = 0.0001
        self.power = 2
        self.lambda_min = 5.0
        self.margin_formula = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

        # loss args
        self.loss_type = loss_type
        self.lampda = lampda
        self.gamma = gamma
        self.save_rate = save_rate

    def forward(self, embedding: torch.Tensor, ground_truth):
        norm_weight = F.normalize(self.weight)
        norm_embd = F.normalize(embedding)
        cos_theta = F.linear(norm_embd, norm_weight).clamp(-1 + 1e-7, 1 - 1e-7)  # for numerical stability
        gt = torch.gather(cos_theta, 1, ground_truth.view(-1, 1))  # ground truth score

        if self.fc_type == 'FC':
            final_gt = gt
        elif self.fc_type == 'SphereFace':
            self.iter += 1
            self.cur_lambda = max(self.lambda_min, self.base * (1 + self.alpha * self.iter) ** (-1 * self.power))
            phi = self.margin_formula[int(self.margin)](gt)  # cos(margin * gt)
            theta = gt.data.acos()
            k = ((self.margin * theta) / math.pi).floor()
            phi_theta = ((-1.0) ** k) * phi - 2 * k
            final_gt = (self.cur_lambda * gt + phi_theta) / (1 + self.cur_lambda)
        elif self.fc_type == 'AM':  # cosface
            final_gt = gt - self.margin
        elif self.fc_type == 'Arc':  # arcface
            sin_theta = torch.sqrt((1.0 - torch.pow(gt, 2)).clamp(0 + 1e-7, 1 - 1e-7))
            phi = gt * self.cos_m - sin_theta * self.sin_m  # cos(gt + margin)
            if self.easy_margin:
                final_gt = torch.where(gt > 0, phi, gt)
            else:
                final_gt = torch.where(gt > self.th, phi, gt - self.mm)
        elif self.fc_type == 'MV-AM':
            mask = cos_theta > gt - self.margin
            hard_vector = cos_theta[mask]
            cos_theta[mask] = (self.t + 1.0) * hard_vector + self.t  # adaptive
            # cos_theta[mask] = hard_vector + self.t  #fixed
            final_gt = gt - self.margin
        elif self.fc_type == 'MV-Arc':
            sin_theta = torch.sqrt((1.0 - torch.pow(gt, 2)).clamp(0 + 1e-7, 1 - 1e-7))
            phi = gt * self.cos_m - sin_theta * self.sin_m  # cos(gt + margin)

            mask = cos_theta > phi
            hard_vector = cos_theta[mask]
            cos_theta[mask] = (self.t + 1.0) * hard_vector + self.t  # adaptive
            # cos_theta[mask] = hard_vector + self.t #fixed
            if self.easy_margin:
                final_gt = torch.where(gt > 0, phi, gt)
            else:
                final_gt = torch.where(gt > self.th, phi, gt - self.mm)
                # final_gt = torch.where(gt > cos_theta_m, cos_theta_m, gt)
        else:
            raise Exception('unknown fc type!')

        cos_theta = torch.scatter(cos_theta, 1, ground_truth.view(-1, 1), final_gt)
        cos_theta = self.scale * cos_theta

        if 'Softmax' in self.loss_type.split("-"):
            loss_final = F.cross_entropy(cos_theta, ground_truth)
        elif 'FocalLoss' in self.loss_type.split("-"):
            assert (self.gamma >= 0)
            input = F.cross_entropy(cos_theta, ground_truth, reduce=False)
            pt = torch.exp(-input)
            loss = (1 - pt) ** self.gamma * input
            loss_final = loss.mean()
        elif 'HardMining' in self.loss_type.split("-"):
            batch_size = cos_theta.shape[0]
            loss = F.cross_entropy(cos_theta, ground_truth, reduction='none')
            ind_sorted = torch.argsort(-loss)  # from big to small
            num_saved = int(self.save_rate * batch_size)
            ind_update = ind_sorted[:num_saved]
            loss_final = torch.sum(F.cross_entropy(cos_theta[ind_update], ground_truth[ind_update]))
        else:
            raise Exception('unknown loss type!!')
        if 'Disc' in self.loss_type.split("-"):
            epsilon = norm_embd - norm_weight[ground_truth]
            norm_bias = self.xi_bias.norm().clamp(0, 0.05)
            disc_loss = (epsilon - F.normalize(self.xi_bias) * norm_bias).norm().mean()
            return loss_final + self.lampda * disc_loss
        return loss_final
