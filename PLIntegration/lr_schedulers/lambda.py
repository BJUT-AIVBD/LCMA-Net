#!usr/bin/env python
# -*- coding:utf-8 _*-
import torch
import inspect
import sys

def insightface_webface_scheduler(step, **kwargs):
    if step <= 20e3:
        return 1
    elif 20e3 < step <= 28e3:
        return 0.1
    else:
        return 0.01


def insightface_ms1m_scheduler(step, **kwargs):
    if step <= 100e3:
        return 1
    elif 100e3 < step <= 160e3:
        return 0.1
    else:
        return 0.01


def svsoftmax_webface_scheduler(step, **kwargs):
    if step <= 4e3:
        return 1
    elif 4e3 < step <= 10e3:
        return 0.1
    elif 10e3 < step <= 18e3:
        return 0.01
    else:
        return 0.001


def keras_lr_decay(step, decay=0.0001, **kwargs):
    return 1. / (1. + decay * step)


def get_scheduler(opt, name, last_epoch, **kwargs):
    return torch.optim.lr_scheduler.LambdaLR(optimizer=opt,
                                             lr_lambda=lambda step: getattr(sys.modules[__name__],
                                                                            name)(step, **kwargs),
                                             last_epoch=last_epoch - 1)
