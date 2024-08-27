#!usr/bin/env python
# -*- coding:utf-8 _*-
import importlib
from warnings import simplefilter

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.warnings import LightningDeprecationWarning

simplefilter("ignore", UserWarning)
simplefilter("ignore", LightningDeprecationWarning)

OmegaConf.register_new_resolver("get_module", lambda x: x.split('.')[-1])
OmegaConf.register_new_resolver("get_name", lambda x: x.split('/')[-1].split(".")[0])
OmegaConf.register_new_resolver("get_ckpt_modules", lambda x: x.split("/")[-6])
OmegaConf.register_new_resolver("get_ckpt_epoch", lambda x:x.split("/")[-1].split("-")[0].split("=")[-1])

@hydra.main(config_path='PLIntegration/conf', config_name='config')
def main(cfg: DictConfig):
    trainer = importlib.import_module("PLIntegration.experiments.{}".format(cfg.exp_name)).__getattribute__("train")
    trainer(cfg.experiments)


if __name__ == '__main__':
    main()
