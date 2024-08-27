#!usr/bin/env python
# -*- coding:utf-8 _*-
import importlib

from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision.transforms import Compose


def get_dataloaders(cfg: DictConfig):
    train_dls = []
    for ds_i, ds_target in enumerate(cfg.__target__):
        __D_pkg__ = ".".join(ds_target.split(".")[:-1])
        __D_name__ = ds_target.split(".")[-1]
        __D__ = importlib.import_module(__D_pkg__).__getattribute__(__D_name__)

        transformers = []
        transformer = None
        if cfg.args[ds_i].get("transformers", None) is not None:
            for i, target in enumerate(cfg.args[ds_i].transformers.__target__):
                __T_pkg__ = ".".join(target.split(".")[:-1])
                __T_name__ = target.split(".")[-1]
                __T__ = importlib.import_module(__T_pkg__).__getattribute__(__T_name__)
                if cfg.args[ds_i].transformers.args[i] is None:
                    transformers.append(__T__())
                else:
                    transformers.append(__T__(**cfg.args[ds_i].transformers.args[i]))
            transformer = Compose(transformers)
        train_ds = __D__(**cfg.args[ds_i], **{'transformer': transformer})
        train_dls.append(DataLoader(train_ds, **cfg.args[ds_i].loader))
    return train_dls
