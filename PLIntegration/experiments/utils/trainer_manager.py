#!usr/bin/env python
# -*- coding:utf-8 _*-
import importlib
import os

from omegaconf import DictConfig


def get_callbacks(cfg: DictConfig):
    callbacks = []
    for i, target in enumerate(cfg.__target__):
        __T_pkg__ = ".".join(target.split(".")[:-1])
        __T_name__ = target.split(".")[-1]
        __T__ = importlib.import_module(__T_pkg__).__getattribute__(__T_name__)
        if cfg.args[i] is None:
            callbacks.append(__T__())
        else:
            callbacks.append(__T__(**cfg.args[i]))
    return callbacks


def get_loggers(cfg: DictConfig):
    loggers = []
    for i, target in enumerate(cfg.__target__):
        # cfg.args[i].save_dir = os.path.join(os.getcwd(), cfg.args[i].save_dir)
        __T_pkg__ = ".".join(target.split(".")[:-1])
        __T_name__ = target.split(".")[-1]
        __T__ = importlib.import_module(__T_pkg__).__getattribute__(__T_name__)
        if cfg.args[i] is None:
            loggers.append(__T__())
        else:
            loggers.append(__T__(**cfg.args[i]))
    return loggers
