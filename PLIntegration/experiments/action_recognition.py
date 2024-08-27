#!usr/bin/env python
# -*- coding:utf-8 _*-
import shutil

import pytorch_lightning as pl
import torch.onnx
from omegaconf import DictConfig

from PLIntegration.experiments.utils import trainer_manager
from PLIntegration.experiments import pl_models

pl.seed_everything(970614)


def train(cfg: DictConfig):
    data_module = pl_models.VideoClassificationData(cfg.model)
    model = pl_models.VideoClassification(cfg.model)
    callbacks = trainer_manager.get_callbacks(cfg.callbacks)
    loggers = trainer_manager.get_loggers(cfg.loggers)
    trainer = pl.Trainer(**cfg.trainer, **{"callbacks": callbacks, "logger": loggers})
    trainer.fit(model, datamodule=data_module)


def test(cfg: DictConfig):
    data_module = pl_models.VideoClassificationData(cfg.model)
    model = pl_models.VideoClassification(cfg.model)
    loggers = trainer_manager.get_loggers(cfg.loggers)
    trainer = pl.Trainer(**cfg.trainer, **{"logger": loggers})
    trainer.test(model, ckpt_path=cfg.checkpoint, datamodule=data_module)
    torch.onnx.export(model,
                      torch.rand(2, cfg.model.feature_extractor.input_channels, 16, cfg.image_size, cfg.image_size),
                      "model.onnx")
    shutil.copy(cfg.checkpoint, ".")
    shutil.copy(cfg.model.trial_path, ".")
