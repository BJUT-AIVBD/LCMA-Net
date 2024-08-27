#!usr/bin/env python
# -*- coding:utf-8 _*-
import torch
from time import sleep
import pytorch_lightning as pl
from omegaconf import OmegaConf

from PLIntegration.experiments.mma_reid import MFFN_DBFF, MFFN_HAFM, MFFN_ICODE1, MFFN_Transformer, ML_MDA, \
    MMClassification_XY1, \
    MMClassification_XY_YZ1, \
    MMClassification_XY3, \
    MMClassification_XZ1, \
    MMClassification_XZ_YZ1, MMClassification_YZ1, MFFN_MLB1, MFFN_attVALD1
from PLIntegration.experiments.pl_models import FaceClassification, VideoClassification, MMClassification
from PLIntegration.networks._2d.seqface import SeqFace21

cfg = "/mnt/homeold/yjc/PythonProject/PLIntegration/train_outputs/mma_reid/MFFN_HAFM/2024-08-03_17-33-14/.hydra/config.yaml"
ckpt = "/mnt/homeold/yjc/PythonProject/PLIntegration/train_outputs/mma_reid/MFFN_HAFM/2024-08-03_17-33-14/checkpoints/last.ckpt"
input_shape = [(2, 3, 112, 112), (2, 3, 16, 112, 112), (2, 59049)]

input_t = [torch.rand(input_shape[0]).to('cuda:0'),
           torch.rand(input_shape[1]).to('cuda:0'),
           torch.rand(input_shape[2]).to('cuda:0')]
# input_t = [[
#     torch.rand((2, 3, 112, 112)).to('cuda:0'),
#     torch.rand((2, 3, 112, 112)).to('cuda:0'),
#     torch.rand((2, 3, 257, 231)).to('cuda:0'),
# ]]
# input_t = torch.rand(input_shape[0]).to('cuda:0')
model = MFFN_HAFM.load_from_checkpoint(ckpt,
                                    args=OmegaConf.load(cfg).experiments.model,
                                    strict=False)
model = model.to("cuda:0")
model.eval()
model.freeze()
model.requires_grad_(False)


def speed_test():
    model(input_t)

if __name__ == '__main__':
    import timeit

    print("start test")
    time_cost = timeit.repeat(stmt="speed_test()", setup="from __main__ import speed_test", number=100, repeat=10)
    print("time_cost = ", min(time_cost) * 10)