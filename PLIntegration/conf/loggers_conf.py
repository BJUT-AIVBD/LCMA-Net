#!usr/bin/env python
# -*- coding:utf-8 _*-

from dataclasses import dataclass, MISSING, field
from typing import List, Dict, Optional, Any

import pytorch_lightning.loggers


@dataclass
class log_conf_unit:
    args: Optional[Dict[Any, Any]] = MISSING
    __target__: str = MISSING


@dataclass
class CometLogger(log_conf_unit):
    args: Optional[Dict[Any, Any]] = field(default_factory=lambda: {})
    __target__: str = "pytorch_lightning.loggers.CometLogger"


@dataclass
class MLFlowLogger(log_conf_unit):
    args: Optional[Dict[Any, Any]] = field(default_factory=lambda: {})
    __target__: str = "pytorch_lightning.loggers.MLFlowLogger"


@dataclass
class NeptuneLogger(log_conf_unit):
    args: Optional[Dict[Any, Any]] = field(default_factory=lambda: {})
    __target__: str = "pytorch_lightning.loggers.NeptuneLogger"


@dataclass
class TensorBoardLogger(log_conf_unit):
    args: Optional[Dict[Any, Any]] = field(default_factory=lambda: {"save_dir": "tb_logs",
                                                                    "name": "",
                                                                    "version": "",
                                                                    "log_graph": True,
                                                                    "default_hp_metric": True,
                                                                    "prefix": "",
                                                                    "sub_dir": None})
    __target__: str = "pytorch_lightning.loggers.TensorBoardLogger"


@dataclass
class WandbLogger(log_conf_unit):
    args: Optional[Dict[Any, Any]] = field(default_factory=lambda: {})
    __target__: str = "pytorch_lightning.loggers.WandbLogger"
