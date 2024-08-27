#!usr/bin/env python
# -*- coding:utf-8 _*-

from dataclasses import dataclass, MISSING, field
from typing import Dict, Optional, Any


@dataclass
class cb_conf_unit:
    args: Optional[Dict[Any, Any]] = MISSING
    __target__: str = MISSING


@dataclass
class RichModelSummary(cb_conf_unit):
    __target__: str = "pytorch_lightning.callbacks.RichModelSummary"
    args: Optional[Dict[Any, Any]] = field(default_factory=lambda: {"max_depth": 2})


@dataclass
class CustomProgressBar(cb_conf_unit):
    __target__: str = "pytorch_lightning.callbacks.CustomProgressBar"
    args: Optional[Dict[Any, Any]] = field(default_factory=lambda: {"refresh_rate": 1,
                                                                    "position": 0})


@dataclass
class DeviceStatsMonitor(cb_conf_unit):
    __target__: str = "pytorch_lightning.callbacks.DeviceStatsMonitor"
    args: Optional[Dict[Any, Any]] = None


@dataclass
class LearningRateMonitor(cb_conf_unit):
    __target__: str = "pytorch_lightning.callbacks.LearningRateMonitor"
    args: Optional[Dict[Any, Any]] = None


@dataclass
class ModelCheckpoint(cb_conf_unit):
    __target__: str = "pytorch_lightning.callbacks.ModelCheckpoint"
    args: Optional[Dict[Any, Any]] = field(default_factory=lambda: {"dirpath": "checkpoints",
                                                                    "filename": "{epoch}-{train_loss:.2f}-{EER:.2f}",
                                                                    "monitor": "EER",
                                                                    "verbose": True,
                                                                    "save_top_k": 3,
                                                                    "save_last": True,
                                                                    "mode": "min",
                                                                    "auto_insert_metric_name": True,
                                                                    "every_n_epochs": None})


cb_default_enable = [RichModelSummary(), CustomProgressBar(), LearningRateMonitor(), ModelCheckpoint()]
cb_all_enable = [RichModelSummary(), CustomProgressBar(), DeviceStatsMonitor(), LearningRateMonitor(),
                 ModelCheckpoint()]
