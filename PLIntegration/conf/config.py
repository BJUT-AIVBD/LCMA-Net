#!usr/bin/env python
# -*- coding:utf-8 _*-

from PLIntegration.conf.hydra.hydra import *
from PLIntegration.conf.experiments import *
from hydra.core.config_store import ConfigStore

from PLIntegration.conf.experiments.face_verification.exp import *
from PLIntegration.conf.hydra.hydra import *

defualts = [
    {"exp_name": "face_verification"},
    {"hydra": "loss"},
    {"experiments": "exp"}
]


@dataclass
class Base:
    exp_name: Any = MISSING
    hydra: Any = MISSING
    experiments: Any = MISSING

    defualts: List[Any] = field(default_factory=lambda: defualts)



cs = ConfigStore.instance()
cs.store(group="hydra/hydra", name="loss", node=loss_test)
cs.store(group="hydra/hydra", name="aggregator_test", node=aggregator_test)
cs.store(group="experiments", name="exp", node=exp_config)
cs.store(name="test", node=Base)
