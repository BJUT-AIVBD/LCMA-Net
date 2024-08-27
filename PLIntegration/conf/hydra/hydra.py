#!usr/bin/env python
# -*- coding:utf-8 _*-

from dataclasses import dataclass

from hydra.conf import HydraConf, RunDir


@dataclass
class loss_test(HydraConf):
    run: RunDir = RunDir("${hydra.job.name}_outputs/${exp_name}/${experiments.name}/"
                         
                         "${now:%Y-%m-%d_%H-%M-%S}")


@dataclass
class aggregator_test(HydraConf):
    run: RunDir = RunDir("${hydra.job.name}_outputs/${exp_name}/${experiments.name}/"
                        
                         "${now:%Y-%m-%d_%H-%M-%S}")
