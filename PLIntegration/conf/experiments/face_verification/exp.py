#!usr/bin/env python
# -*- coding:utf-8 _*-

from PLIntegration.conf.callbacks_conf import *
from PLIntegration.conf.loggers_conf import *


@dataclass
class exp_config:
    name: str = "PolyNet"
    device: str = "cuda"
    image_size: int = 112
    batch_size: int = 64
    learning_rate: float = 0.1
    accumulate_grad_batches: int = 1
    checkpoint: Optional[str] = None
    fast_dev_run: int = 0
    overfit_batches: int = 0
    callbacks: List[cb_conf_unit] = field(default_factory=lambda: cb_default_enable)
    loggers: List[log_conf_unit] = field(default_factory=lambda: [TensorBoardLogger()])
    model = MISSING
    trainer = MISSING

