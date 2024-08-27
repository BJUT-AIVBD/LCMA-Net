import shutil

import omegaconf
import pytorch_lightning as pl
import torch.onnx
from omegaconf import DictConfig, OmegaConf

from PLIntegration.experiments.utils import trainer_manager
from PLIntegration.experiments import pl_models

pl.seed_everything(970614, True)


def train(cfg: DictConfig):
    data_module = pl_models.MMClassificationData(cfg.model)
    model = pl_models.MMClassification(cfg.model)
    callbacks = trainer_manager.get_callbacks(cfg.callbacks)
    loggers = trainer_manager.get_loggers(cfg.loggers)
    trainer = pl.Trainer(**cfg.trainer, **{"callbacks": callbacks, "logger": loggers})
    trainer.fit(model, datamodule=data_module)


def test(cfg: DictConfig):
    data_module = pl_models.MMClassificationData(OmegaConf.load(cfg.model.config).experiments.model)
    model = pl_models.MMClassification.load_from_checkpoint(cfg.model.checkpoint,
                                                            args=OmegaConf.load(cfg.model.config).experiments.model)
    loggers = trainer_manager.get_loggers(cfg.loggers)
    trainer = pl.Trainer(**cfg.trainer, **{"logger": loggers})
    trainer.test(model, ckpt_path=cfg.checkpoint, datamodule=data_module)
    torch.onnx.export(model,
                      (torch.rand((2, 3, 112, 112)).to("cuda"),
                       torch.rand((2, 3, 16, 112, 112)).to("cuda")),
                      "model.onnx")
    if cfg.checkpoint is not None:
        shutil.copy(cfg.checkpoint, ".")
    if cfg.model.trial_path is not None:
        shutil.copy(cfg.model.trial_path, ".")
