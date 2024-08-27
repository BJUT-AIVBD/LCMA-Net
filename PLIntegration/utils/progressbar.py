#!usr/bin/env python
# -*- coding:utf-8 _*-

from typing import Dict, Union
import pytorch_lightning as pl

from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar, RichProgressBarTheme


class CustomRichProgressBar(RichProgressBar):
    def __init__(self, refresh_rate_per_second: int = 10,
                 leave: bool = False,
                 theme: RichProgressBarTheme = RichProgressBarTheme()):
        super().__init__(refresh_rate_per_second, leave, theme)

    def get_metrics(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> Dict[str, Union[int, str]]:
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return items


class CustomProgressBar(TQDMProgressBar):
    def __init__(self, refresh_rate: int = 1, position: int = 0):
        super().__init__(refresh_rate, position)

    def get_metrics(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> Dict[str, Union[int, str]]:
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return items
