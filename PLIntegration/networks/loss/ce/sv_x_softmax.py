#!usr/bin/env python
# -*- coding:utf-8 _*-
import math

import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter


class SVSoftmax(Module):
    def __init__(self, fc_type='MV-AM', loss_type="Softmax",
                 margin=0.35, t=0.2, scale=32,
                 gamma=2.0, save_rate=0.9,
                 embed_size=512, num_classes=72690,
                 easy_margin=True, **kwargs):
        super().__init__()
        self.weight = Parameter(torch.Tensor(embed_size, num_classes))
        # initial kernel
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.margin = margin
        self.t = t
        self.easy_margin = easy_margin
        self.scale = scale
        self.fc_type = fc_type
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        # duplication formula
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
        self.gamma = gamma
        self.save_rate = save_rate

    def forward(self, x, label):  # x (M, K), w(K, N), y = xw (M, N), note both x and w are already l2 normalized.
        kernel_norm = F.normalize(self.weight, dim=0)
        cos_theta = torch.mm(x, kernel_norm)
        cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)  # for numerical stability
        batch_size = label.size(0)
        gt = cos_theta[torch.arange(0, batch_size), label].view(-1, 1)  # ground truth score

        if self.fc_type == 'FC':
            final_gt = gt
        elif self.fc_type == 'SphereFace':
            self.iter += 1
            self.cur_lambda = max(self.lambda_min, self.base * (1 + self.alpha * self.iter) ** (-1 * self.power))
            cos_theta_m = self.margin_formula[int(self.margin)](gt)  # cos(margin * gt)
            theta = gt.data.acos()
            k = ((self.margin * theta) / math.pi).floor()
            phi_theta = ((-1.0) ** k) * cos_theta_m - 2 * k
            final_gt = (self.cur_lambda * gt + phi_theta) / (1 + self.cur_lambda)
        elif self.fc_type == 'AM':  # cosface
            final_gt = gt - self.margin
        elif self.fc_type == 'Arc':  # arcface
            sin_theta = torch.sqrt((1.0 - torch.pow(cos_theta, 2)).clamp(0 + 1e-7, 1 - 1e-7))
            cos_theta_m = gt * self.cos_m - sin_theta * self.sin_m  # cos(gt + margin)
            if self.easy_margin:
                final_gt = torch.where(gt > 0, cos_theta_m, gt)
            else:
                final_gt = torch.where(gt > self.th, cos_theta_m, gt - self.mm)
        elif self.fc_type == 'MV-AM':
            mask = cos_theta > gt - self.margin
            hard_vector = cos_theta[mask]
            cos_theta[mask] = (self.t + 1.0) * hard_vector + self.t  # adaptive
            # cos_theta[mask] = hard_vector + self.t  #fixed
            if self.easy_margin:
                final_gt = torch.where(gt > 0, gt - self.margin, gt)
            else:
                final_gt = gt - self.margin
        elif self.fc_type == 'MV-Arc':
            sin_theta = torch.sqrt(1.0 - torch.pow(gt, 2))
            cos_theta_m = gt * self.cos_m - sin_theta * self.sin_m  # cos(gt + margin)

            mask = cos_theta > cos_theta_m
            hard_vector = cos_theta[mask]
            cos_theta[mask] = (self.t + 1.0) * hard_vector + self.t  # adaptive
            # cos_theta[mask] = hard_vector + self.t #fixed
            if self.easy_margin:
                final_gt = torch.where(gt > 0, cos_theta_m, gt)
            else:
                final_gt = cos_theta_m
                # final_gt = torch.where(gt > cos_theta_m, cos_theta_m, gt)
        else:
            raise Exception('unknown fc type!')

        cos_theta.scatter_(1, label.view(-1, 1), final_gt)
        cos_theta *= self.scale

        if self.loss_type == 'Softmax':
            loss_final = F.cross_entropy(cos_theta, label)
        elif self.loss_type == 'FocalLoss':
            assert (self.gamma >= 0)
            input = F.cross_entropy(cos_theta, label, reduce=False)
            pt = torch.exp(-input)
            loss = (1 - pt) ** self.gamma * input
            loss_final = loss.mean()
        elif self.loss_type == 'HardMining':
            batch_size = cos_theta.shape[0]
            loss = F.cross_entropy(cos_theta, label, reduction='none')
            ind_sorted = torch.argsort(-loss)  # from big to small
            num_saved = int(self.save_rate * batch_size)
            ind_update = ind_sorted[:num_saved]
            loss_final = torch.sum(F.cross_entropy(cos_theta[ind_update], label[ind_update]))
        else:
            raise Exception('unknown loss type!!')

        return loss_final
