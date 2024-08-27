#!usr/bin/env python
# -*- coding:utf-8 _*-
import importlib
import shutil
from functools import partial
from typing import Any, List

import pytorch_lightning as pl
import torch.optim.sgd
from omegaconf import DictConfig, OmegaConf, ListConfig
from pytorch_lightning.utilities.model_summary import summarize, _format_summary_table
from torch import nn
from torch.nn.functional import normalize
from torchvision.models import resnet18
from tqdm import tqdm
import torch.nn.functional as F

from PLIntegration.experiments import pl_models
from PLIntegration.experiments.utils import trainer_manager
from PLIntegration.metrics.verification import *
from PLIntegration.networks._2d.polynet import PolyNet, PolyBlock
from PLIntegration.networks._2d.rawnet import RawNet_NL_GRU, LayerNorm, SincConv_fast, Residual_block_wFRM, \
    Residual_block_NL
from PLIntegration.networks._3d.resnet import STDAResNeXt3D, ResNeXtBottleneck3D, STDABottleneck, conv1x1x1
from PLIntegration.networks.aggregators.VLAD import AttVlad
from PLIntegration.networks.mixer.MLMA import MLMA
from PLIntegration.networks.mixer.compact_bilinear_pooling import CompactBilinearPooling


class MMARawNet(RawNet_NL_GRU):
    def __init__(self, nl_idim=256, nl_odim=64, **kwargs):
        super().__init__(nl_idim, nl_odim, **kwargs)

    def forward(self, x):
        nb_samp = x.shape[0]
        len_seq = x.shape[1]
        x = self.ln(x)
        x = x.view(nb_samp, 1, len_seq)
        x = F.max_pool1d(torch.abs(self.first_conv(x)), 3)
        x = self.first_bn(x)
        x = self.lrelu_keras(x)

        x = self.block0(x)
        x = self.block1(x)

        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.bn_before_gru(x)
        x = self.lrelu_keras(x)
        x = x.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:, -1, :]
        code = self.fc1_gru(x)

        return code


class MMAFace(PolyNet):
    def __init__(self, input_channels: int, filts: List[int], layers: List[int], dropout: float = 0, **kwargs):
        super().__init__(input_channels, filts, layers, dropout, **kwargs)


class MMAMotion(STDAResNeXt3D):
    def __init__(self, layers=None, **kwargs):
        if layers is None:
            layers = [3, 4, 23, 3]
        super().__init__(layers, **kwargs)


# region    # <--- LCMPA-Single ---> #

class MMClassification_XY1(pl.LightningModule):
    def __init__(self, args: DictConfig):
        super().__init__()
        self.example_input_array = [[torch.rand((2, 3, args.image_size, args.image_size)).to(args.device),
                                     torch.rand((2, 3, 16, args.image_size, args.image_size)).to(args.device),
                                     torch.rand((2, args.nb_samp)).to(args.device)]]
        self.args = args
        self.batch_size = self.args.batch_size
        self.learning_rate = self.args.learning_rate

        # loading Feature Extractors
        # region    # <--- RawNet ---> #
        self.voice_ln = LayerNorm(59049)
        self.voice_first_conv = SincConv_fast(in_channels=1,
                                              out_channels=128,
                                              kernel_size=251)

        self.voice_first_bn = nn.BatchNorm1d(num_features=128)
        self.voice_lrelu = nn.LeakyReLU()
        self.voice_lrelu_keras = nn.LeakyReLU(negative_slope=0.3)

        self.voice_block0 = nn.Sequential(Residual_block_wFRM(nb_filts=[128, 128], first=True))
        self.voice_block1 = nn.Sequential(Residual_block_wFRM(nb_filts=[128, 128]))
        self.voice_block2 = nn.Sequential(Residual_block_wFRM(nb_filts=[128, 256]))
        self.voice_block3 = nn.Sequential(Residual_block_NL(nb_filts=[256, 256, 256, 64]))
        self.voice_block4 = nn.Sequential(Residual_block_NL(nb_filts=[256, 256, 256, 64]))
        self.voice_block5 = nn.Sequential(Residual_block_NL(nb_filts=[256, 256, 256, 64]))

        self.voice_bn_before_gru = nn.BatchNorm1d(num_features=256)
        self.voice_gru = nn.GRU(input_size=256,
                                hidden_size=1024,
                                num_layers=1,
                                bidirectional=False,
                                batch_first=True)
        self.voice_fc1_gru = nn.Linear(in_features=1024,
                                       out_features=1024)
        self.voice_sig = nn.Sigmoid()
        # endregion # <--- RawNet ---> #

        # region    # <--- II-Net ---> #

        self.face_in_channels = 64
        self.face_conv1 = nn.Conv2d(3, 64,
                                    kernel_size=(3, 3), stride=(1, 1),
                                    padding=(1, 1), bias=False)
        self.face_bn1 = nn.BatchNorm2d(64)
        self.face_prelu = nn.PReLU()
        self.face_layer1 = self._make_layer_face(PolyBlock, 64, 3, stride=2)
        self.face_layer2 = self._make_layer_face(PolyBlock, 128, 4, stride=2)
        self.face_layer3 = self._make_layer_face(PolyBlock, 256, 6, stride=2)
        self.face_layer4 = self._make_layer_face(PolyBlock, 512, 3, stride=2)
        self.face_bn2 = nn.BatchNorm2d(512)
        self.face_dropout = nn.Dropout(0)

        # endregion # <--- II-Net ---> #

        # region    # <--- Motion ---> #

        self.video_in_planes = 128
        self.video_cardinality = 32
        self.video_STDA_input_shape = [1024, 2, 7, 7]
        self.video_conv1 = nn.Conv3d(3,
                                     self.video_in_planes,
                                     kernel_size=(7, 7, 7),
                                     stride=(1, 2, 2),
                                     padding=(7 // 2, 3, 3),
                                     bias=False)
        self.video_bn1 = nn.BatchNorm3d(self.video_in_planes)
        self.video_relu = nn.ReLU(inplace=True)
        self.video_maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.video_layer1 = self._make_layer_video(ResNeXtBottleneck3D, 3, 128,
                                                   "B")
        self.video_layer2 = self._make_layer_video(ResNeXtBottleneck3D,
                                                   4,
                                                   256,
                                                   "B",
                                                   stride=2)
        self.video_layer3 = self._make_layer_video(ResNeXtBottleneck3D,
                                                   23,
                                                   512,
                                                   "B",
                                                   stride=2)
        self.video_layer4 = self._make_layer_video(STDABottleneck,
                                                   3,
                                                   1024,
                                                   "B",
                                                   stride=2)

        # endregion # <--- Motion ---> #

        self.attxy_voice = nn.Conv1d(80, 1, 1)
        self.attxy_face = nn.Conv1d(14 * 14, 1, 1)
        self.attxy_expand = nn.Conv1d(1, 80, 1)
        self.attxy_bn = nn.BatchNorm1d(256)
        self.attyx_voice = nn.Conv1d(80, 1, 1)
        self.attyx_face = nn.Conv1d(14 * 14, 1, 1)
        self.attyx_expand = nn.Conv1d(1, 14 * 14, 1)
        self.attyx_bn = nn.BatchNorm2d(256)

        for layer in self.named_modules():
            if "attxy" not in layer[0] and "attyx" not in layer[0] and \
                    "attxz" not in layer[0] and "attzx" not in layer[0] and \
                    "attyz" not in layer[0] and "attyz" not in layer[0]:
                layer[1].eval().requires_grad_(False)
        for layer in self.named_modules():
            if "all" in args.grad:
                layer[1].train().requires_grad_()
            if layer[0] in args.grad:
                layer[1].train().requires_grad_()
            elif "layer" in layer[0] or "block" in layer[0]:
                if ".".join(layer[0].split(".")[:-1]) in args.grad:
                    layer[1].train().requires_grad_()
                if ".".join(layer[0].split(".")[:-2]) in args.grad:
                    layer[1].train().requires_grad_()

        # loading Embedding Aggregators
        if isinstance(args.embedding_aggregator.__target__, ListConfig):
            self.embedding_aggregator = []
            for i in range(len(args.embedding_aggregator.__target__)):
                __E_pkg__ = ".".join(args.embedding_aggregator.__target__[i].split(".")[:-1])
                __E_name__ = args.embedding_aggregator.__target__[i].split(".")[-1]
                __E__ = importlib.import_module(__E_pkg__).__getattribute__(__E_name__)
                if issubclass(__E__, pl.LightningModule):
                    embedding_aggregator = __E__.load_from_checkpoint(args.embedding_aggregator.args[i].checkpoint,
                                                                      args=OmegaConf.load(
                                                                          args.embedding_aggregator.args[
                                                                              i].cfg).experiments.model)
                    embedding_aggregator = embedding_aggregator.embedding_aggregator
                    embedding_aggregator.eval().requires_grad_(False)
                    if args.embedding_aggregator.args[i].grad[0] == "all":
                        embedding_aggregator.train().requires_grad_(True)
                    for layer in embedding_aggregator.named_modules():
                        if layer[0] in args.embedding_aggregator.args[i].grad:
                            layer[1].train().requires_grad_()
                    self.embedding_aggregator.append(embedding_aggregator)
                else:
                    self.embedding_aggregator.append(__E__(**args.embedding_aggregator.args[i]))
                    ckpt = args.embedding_aggregator.args[i].get("checkpoint", None)
                    if ckpt is not None:
                        try:
                            self.embedding_aggregator[i].load_state_dict(torch.load(ckpt)["state_dict"], strict=False)
                        except:
                            self.embedding_aggregator[i].load_state_dict(torch.load(ckpt), strict=False)

            self.embedding_aggregator = nn.ModuleList(self.embedding_aggregator)
        else:
            print("embedding_aggregator.__target__ should be a list")
            raise TypeError

        # loading Embedding Mixer
        __M_pkg__ = ".".join(args.embedding_mixer.__target__.split(".")[:-1])
        __M_name__ = args.embedding_mixer.__target__.split(".")[-1]
        __M__ = importlib.import_module(__M_pkg__).__getattribute__(__M_name__)
        self.embedding_mixer = __M__(**args.embedding_mixer)

        # loading Loss Function & Classifier
        __L_pkg__ = ".".join(args.loss_function.__target__.split(".")[:-1])
        __L_name__ = args.loss_function.__target__.split(".")[-1]
        __L__ = importlib.import_module(__L_pkg__).__getattribute__(__L_name__)
        self.loss_function = __L__(**args.loss_function)

        # loading Optimizer
        __O_pkg__ = ".".join(args.optimizer.__target__.split(".")[:-1])
        __O_name__ = args.optimizer.__target__.split(".")[-1]
        __O__ = importlib.import_module(__O_pkg__).__getattribute__(__O_name__)
        self.optim = __O__

        # loading Learning Rate Scheduler
        __S_pkg__ = ".".join(args.optimizer.lr_scheduler.__target__.split(".")[:-1])
        __S_name__ = args.optimizer.lr_scheduler.__target__.split(".")[-1]
        __S__ = importlib.import_module(__S_pkg__).__getattribute__(__S_name__)
        self.lr_sh = __S__

        try:
            if args.state_dict.split('.')[-1] == "ckpt":
                self.load_state_dict(torch.load(args.state_dict)["state_dict"])
            else:
                self.load_state_dict(torch.load(args.state_dict), strict=False)
        except Exception:
            print("XY1 load ckpt failed")
        self.count = 0

    def _video_downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4), device=self.args.device)
        out = torch.cat([out.data, zero_pads], dim=1)
        return out

    def _make_layer_face(self, block, out_channels, blocks, stride=(1, 1)):
        layers = [block(self.face_in_channels, out_channels, stride, downsample=True)]
        self.face_in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(self.face_in_channels, self.face_in_channels))
        return nn.Sequential(*layers)

    def _make_layer_video(self, block, blocks, planes, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.video_in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._video_downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.video_in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(inplanes=self.video_in_planes,
                  planes=planes,
                  stride=stride,
                  cardinality=self.video_cardinality,
                  downsample=downsample,
                  input_shape=self.video_STDA_input_shape))
        self.video_in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes=self.video_in_planes,
                                planes=planes,
                                stride=1,
                                cardinality=self.video_cardinality,
                                input_shape=[self.video_STDA_input_shape[0] * block.expansion,
                                             int(np.ceil(self.video_STDA_input_shape[1] / stride)),
                                             int(np.ceil(self.video_STDA_input_shape[2] / stride)),
                                             int(np.ceil(self.video_STDA_input_shape[3] / stride))]))

        return nn.Sequential(*layers)

    def forward(self, inpt, **kwargs) -> Any:
        # region    # <--- RawNet ---> #
        nb_samp = inpt[2].shape[0]
        len_seq = inpt[2].shape[1]
        x = self.voice_ln(inpt[2])
        x = x.view(nb_samp, 1, len_seq)
        x = F.max_pool1d(torch.abs(self.voice_first_conv(x)), 3)
        x = self.voice_first_bn(x)
        x = self.voice_lrelu_keras(x)
        x = self.voice_block0(x)
        x = self.voice_block1(x)
        x = self.voice_block2(x)
        x = self.voice_block3(x)
        x = self.voice_block4(x)

        y = self.face_conv1(inpt[0])
        y = self.face_bn1(y)
        y = self.face_prelu(y)
        y = self.face_layer1(y)
        y = self.face_layer2(y)
        y = self.face_layer3(y)

        attxy_voice = self.attxy_voice(x.transpose(1, 2))  # 256*80
        attxy_face = torch.softmax(self.attxy_face(y.view(y.shape[0], y.shape[1], -1).transpose(1, 2)),
                                   -1)  # 256*14*14
        attxy = F.adaptive_avg_pool1d(attxy_voice * attxy_face.transpose(1, 2), 1)  # 256*1
        attxy_expand = self.attxy_bn(self.attxy_expand(attxy.transpose(1, 2)).transpose(1, 2))  # 256*80
        x = x + attxy_expand
        attyx_voice = torch.softmax(self.attyx_voice(x.transpose(1, 2)), -1)  # 256*80
        attyx_face = self.attyx_face(y.view(y.shape[0], y.shape[1], -1).transpose(1, 2))  # 256*14*14
        attyx = F.adaptive_avg_pool1d(attyx_face * attyx_voice.transpose(1, 2), 1)  # 256*1
        attyx_expand = self.attyx_bn(
            self.attyx_expand(attyx.transpose(1, 2)).transpose(1, 2).view(-1, 256, 14, 14))  # 256*14*14
        y = y + attyx_expand

        x = self.voice_block5(x)
        x = self.voice_bn_before_gru(x)
        x = self.voice_lrelu_keras(x)
        x = x.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
        self.voice_gru.flatten_parameters()
        x, _ = self.voice_gru(x)
        x = x[:, -1, :]
        x = self.voice_fc1_gru(x)

        y = self.face_layer4(y)
        y = self.face_bn2(y)
        y = self.face_dropout(y)
        # endregion # <--- II-Net ---> #

        # region    # <--- Motion ---> #
        z = self.video_conv1(inpt[1])
        z = self.video_bn1(z)
        z = self.video_relu(z)
        z = self.video_maxpool(z)

        z = self.video_layer1(z)
        z = self.video_layer2(z)
        z = self.video_layer3(z)
        z = self.video_layer4(z)
        # endregion # <--- Motion ---> #
        embed_x = self.embedding_aggregator[2](x)
        embed_y = self.embedding_aggregator[0](y)
        embed_z = self.embedding_aggregator[1](z)
        embed = self.embedding_mixer([embed_y, embed_z, embed_x])
        return embed

    def configure_optimizers(self):
        opt = self.optim(self.parameters(), **self.args.optimizer.args)
        if self.args.optimizer.lr_scheduler.args.last_epoch is None:
            if self.args.optimizer.lr_scheduler.interval == "step":
                self.args.optimizer.lr_scheduler.args.last_epoch = self.global_step - 1
            else:
                self.args.optimizer.lr_scheduler.args.last_epoch = self.current_epoch - 1
        lr_sh = self.lr_sh(opt, **self.args.optimizer.lr_scheduler.args)
        return [opt], [{'interval': self.args.optimizer.lr_scheduler.interval, 'scheduler': lr_sh}]

    def training_step(self, batch, batch_idx):
        x, y = batch
        embeddings = self(x)
        loss = self.loss_function(embeddings, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, label, pathes = batch
        embeddings = self(x)
        embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings, pathes

    def validation_epoch_end(self, val_step_outputs: List[Any]) -> None:
        # test REID
        embeds = torch.cat([x[0] for x in val_step_outputs])
        pathes = list(np.concatenate([x[1] for x in val_step_outputs], 0))
        d_embeddings = {}
        if not len(pathes) == len(embeds):
            print(len(pathes), len(embeds))
            exit()
        for k, v in zip(pathes, embeds):
            try:
                key = k[:-10]
            except:
                key = k[0][:-10]
            d_embeddings[key] = torch.squeeze(v).cpu()
        y_score = []  # score for each sample
        y = []  # label for each sample
        with open(self.args.trial_path, 'r') as f:
            l_trial = f.readlines()
        for line in l_trial:
            trg, utt_a, utt_b = line.strip().split(' ')
            y.append(int(trg))
            y_score.append(torch.cosine_similarity(d_embeddings[utt_a[:-10]], d_embeddings[utt_b[:-10]], dim=0))
        y_score = torch.tensor(y_score)
        tuned_threshold, eer, fpr, fnr = tune_threshold_from_score(y_score, y, [1, 0.1])
        fnrs, fprs, thresholds = compute_error_rates(y_score, y)
        min_dcf, min_c_det_threshold = compute_min_dcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        try:
            self.log_dict({"EER": eer, "MinDCF": min_dcf}, on_step=False, on_epoch=True)
        except:
            print("Epoch {}: EER logging failed!".format(self.current_epoch))
        print("epoch {} step {}: EER reaches {}%".format(self.current_epoch, self.global_step, eer))

    def test_step(self, batch, batch_idx):
        self.count += 1
        if self.count == 113:
            print("113")
        x, label, pathes = batch
        union_embeddings = self(x)
        union_embeddings = torch.nn.functional.normalize(union_embeddings)
        return union_embeddings, pathes

    def test_epoch_end(self, outputs: List[Any]) -> None:
        # test REID
        result = ["epoch:{}".format(self.current_epoch),
                  "step:{}".format(self.global_step),
                  "ckpt:{}".format(self.args.checkpoint)]

        union_embeds = torch.cat([x[0] for x in outputs])
        pathes = list(np.concatenate([x[-1] for x in outputs], 0))
        d_embeddings = {}
        if not len(pathes) == len(union_embeds):
            print(len(pathes), len(union_embeds))
            exit()
        for k, v1 in zip(pathes, union_embeds):
            try:
                key = k[:-10]
            except:
                key = k[0][:-10]
            d_embeddings[key] = torch.squeeze(v1).cpu()

        y_score = []  # score for each sample
        y = []  # label for each sample
        with open(self.args.trial_path, 'r') as f:
            l_trial = f.readlines()
        for line in tqdm(l_trial, desc="cal_score"):
            trg, utt_a, utt_b = line.strip().split(' ')
            y.append(int(trg))
            y_score.append(torch.cosine_similarity(d_embeddings[utt_a[:-10]], d_embeddings[utt_b[:-10]], dim=0))
        y_score = torch.tensor(y_score)
        tuned_threshold, eer, fpr, fnr = tune_threshold_from_score(y_score, y, [1, 0.1])
        fnrs, fprs, thresholds = compute_error_rates(y_score, y)
        min_dcf, min_c_det_threshold = compute_min_dcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        self.log_dict({"EER": eer, "MinDCF": min_dcf}, on_step=False, on_epoch=True)
        print("Calculated by compute_min_cost:\nEER = {0}%\nminDCF = {1}\n".format(eer, min_dcf))
        print()
        result.append(
            "Epoch {} step {}:\nEER = {}%\nminDCF = {}".format(self.current_epoch, self.global_step, eer, min_dcf))
        with open("result.txt", 'w') as f:
            f.write("\n".join(result))


class MMClassification_XY2(MMClassification_XY1):
    def __init__(self, args: DictConfig):
        super().__init__(args)

    def forward(self, inpt, **kwargs) -> Any:
        # region    # <--- RawNet ---> #
        nb_samp = inpt[2].shape[0]
        len_seq = inpt[2].shape[1]
        x = self.voice_ln(inpt[2])
        x = x.view(nb_samp, 1, len_seq)
        x = F.max_pool1d(torch.abs(self.voice_first_conv(x)), 3)
        x = self.voice_first_bn(x)
        x = self.voice_lrelu_keras(x)
        x = self.voice_block0(x)
        x = self.voice_block1(x)
        x = self.voice_block2(x)
        x = self.voice_block3(x)
        x = self.voice_block4(x)

        y = self.face_conv1(inpt[0])
        y = self.face_bn1(y)
        y = self.face_prelu(y)
        y = self.face_layer1(y)
        y = self.face_layer2(y)
        y = self.face_layer3(y)

        attxy_voice = self.attyz_face(x.transpose(1, 2))  # 256*80
        attxy_face = torch.sigmoid(self.attyz_video(y.view(y.shape[0], y.shape[1], -1).transpose(1, 2)))  # 256*14*14
        attxy = F.adaptive_avg_pool1d(attxy_voice * attxy_face.transpose(1, 2), 1)  # 256*1
        attxy_expand = self.attyz_bn(self.attyz_expand(attxy.transpose(1, 2)).transpose(1, 2))  # 256*80
        x = x + attxy_expand
        attyx_voice = torch.sigmoid(self.attzy_face(x.transpose(1, 2)))  # 256*80
        attyx_face = self.attzy_video(y.view(y.shape[0], y.shape[1], -1).transpose(1, 2))  # 256*14*14
        attyx = F.adaptive_avg_pool1d(attyx_face * attyx_voice.transpose(1, 2), 1)  # 256*1
        attyx_expand = self.attzy_bn(
            self.attzy_expand(attyx.transpose(1, 2)).transpose(1, 2).view(-1, 256, 14, 14))  # 256*14*14
        y = y + attyx_expand

        x = self.voice_block5(x)
        x = self.voice_bn_before_gru(x)
        x = self.voice_lrelu_keras(x)
        x = x.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
        self.voice_gru.flatten_parameters()
        x, _ = self.voice_gru(x)
        x = x[:, -1, :]
        x = self.voice_fc1_gru(x)

        y = self.face_layer4(y)
        y = self.face_bn2(y)
        y = self.face_dropout(y)
        # endregion # <--- II-Net ---> #

        # region    # <--- Motion ---> #
        z = self.video_conv1(inpt[1])
        z = self.video_bn1(z)
        z = self.video_relu(z)
        z = self.video_maxpool(z)

        z = self.video_layer1(z)
        z = self.video_layer2(z)
        z = self.video_layer3(z)
        z = self.video_layer4(z)
        # endregion # <--- Motion ---> #
        embed_x = self.embedding_aggregator[2](x)
        embed_y = self.embedding_aggregator[0](y)
        embed_z = self.embedding_aggregator[1](z)
        embed = self.embedding_mixer([embed_y, embed_z, embed_x])
        return embed


class MMClassification_XY3(MMClassification_XY1):
    def __init__(self, args: DictConfig):
        super().__init__(args)
        att_channel = args.cross_attention.middle_channels
        self.attxy_voice = nn.Conv1d(80, att_channel, 1)
        self.attxy_face = nn.Conv1d(14 * 14, att_channel, 1)
        self.attxy_expand = nn.Conv1d(1, 80, 1)
        self.attxy_bn = nn.BatchNorm1d(256)
        self.attyx_voice = nn.Conv1d(80, att_channel, 1)
        self.attyx_face = nn.Conv1d(14 * 14, att_channel, 1)
        self.attyx_expand = nn.Conv1d(1, 14 * 14, 1)
        self.attyx_bn = nn.BatchNorm2d(256)
        try:
            if args.state_dict.split('.')[-1] == "ckpt":
                self.load_state_dict(torch.load(args.state_dict)["state_dict"])
            else:
                self.load_state_dict(torch.load(args.state_dict), strict=False)
            print("XY3 load ckpt success")
        except Exception:
            print("XY3 load ckpt failed")

    def forward(self, inpt, **kwargs) -> Any:
        # region    # <--- RawNet ---> #
        nb_samp = inpt[2].shape[0]
        len_seq = inpt[2].shape[1]
        x = self.voice_ln(inpt[2])
        x = x.view(nb_samp, 1, len_seq)
        x = F.max_pool1d(torch.abs(self.voice_first_conv(x)), 3)
        x = self.voice_first_bn(x)
        x = self.voice_lrelu_keras(x)
        x = self.voice_block0(x)
        x = self.voice_block1(x)
        x = self.voice_block2(x)
        x = self.voice_block3(x)
        x = self.voice_block4(x)

        y = self.face_conv1(inpt[0])
        y = self.face_bn1(y)
        y = self.face_prelu(y)
        y = self.face_layer1(y)
        y = self.face_layer2(y)
        y = self.face_layer3(y)

        attxy_voice = self.attxy_voice(x.transpose(1, 2))  # 256*80
        attxy_face = torch.softmax(self.attxy_face(y.view(y.shape[0], y.shape[1], -1).transpose(1, 2)), -1)  # 256*14*14
        attxy = F.adaptive_avg_pool1d(torch.matmul(attxy_face.transpose(1, 2), attxy_voice).transpose(1, 2), 1)  # 256*1
        attxy_expand = self.attxy_bn(self.attxy_expand(attxy.transpose(1, 2)).transpose(1, 2))  # 256*80
        x = x + attxy_expand
        attyx_voice = torch.softmax(self.attyx_voice(x.transpose(1, 2)), -1)  # 256*80
        attyx_face = self.attyx_face(y.view(y.shape[0], y.shape[1], -1).transpose(1, 2))  # 256*14*14
        attyx = F.adaptive_avg_pool1d(torch.matmul(attyx_voice.transpose(1, 2), attyx_face).transpose(1, 2), 1)  # 256*1
        attyx_expand = self.attyx_bn(
            self.attyx_expand(attyx.transpose(1, 2)).transpose(1, 2).view(-1, 256, 14, 14))  # 256*14*14
        y = y + attyx_expand

        x = self.voice_block5(x)
        x = self.voice_bn_before_gru(x)
        x = self.voice_lrelu_keras(x)
        x = x.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
        self.voice_gru.flatten_parameters()
        x, _ = self.voice_gru(x)
        x = x[:, -1, :]
        x = self.voice_fc1_gru(x)

        y = self.face_layer4(y)
        y = self.face_bn2(y)
        y = self.face_dropout(y)
        # endregion # <--- II-Net ---> #

        # region    # <--- Motion ---> #
        z = self.video_conv1(inpt[1])
        z = self.video_bn1(z)
        z = self.video_relu(z)
        z = self.video_maxpool(z)

        z = self.video_layer1(z)
        z = self.video_layer2(z)
        z = self.video_layer3(z)
        z = self.video_layer4(z)
        # endregion # <--- Motion ---> #
        embed_x = self.embedding_aggregator[2](x)
        embed_y = self.embedding_aggregator[0](y)
        embed_z = self.embedding_aggregator[1](z)
        embed = self.embedding_mixer([embed_y, embed_z, embed_x])
        return embed


class MMClassification_XY4(MMClassification_XY1):
    def __init__(self, args: DictConfig):
        super().__init__(args)
        att_channel = args.cross_attention.middle_channels
        self.attxy_voice = nn.Conv1d(80, att_channel, 1)
        self.attxy_face = nn.Conv1d(14 * 14, att_channel, 1)
        self.attxy_expand = nn.Conv1d(1, 80, 1)
        self.attxy_bn = nn.BatchNorm1d(256)
        self.attyx_voice = nn.Conv1d(80, att_channel, 1)
        self.attyx_face = nn.Conv1d(14 * 14, att_channel, 1)
        self.attyx_expand = nn.Conv1d(1, 14 * 14, 1)
        self.attyx_bn = nn.BatchNorm2d(256)
        try:
            if args.state_dict.split('.')[-1] == "ckpt":
                self.load_state_dict(torch.load(args.state_dict)["state_dict"])
            else:
                self.load_state_dict(torch.load(args.state_dict), strict=False)
            print("XY4 load ckpt success")
        except Exception:
            print("XY4 load ckpt failed")

    def forward(self, inpt, **kwargs) -> Any:
        # region    # <--- RawNet ---> #
        nb_samp = inpt[2].shape[0]
        len_seq = inpt[2].shape[1]
        x = self.voice_ln(inpt[2])
        x = x.view(nb_samp, 1, len_seq)
        x = F.max_pool1d(torch.abs(self.voice_first_conv(x)), 3)
        x = self.voice_first_bn(x)
        x = self.voice_lrelu_keras(x)
        x = self.voice_block0(x)
        x = self.voice_block1(x)
        x = self.voice_block2(x)
        x = self.voice_block3(x)
        x = self.voice_block4(x)

        y = self.face_conv1(inpt[0])
        y = self.face_bn1(y)
        y = self.face_prelu(y)
        y = self.face_layer1(y)
        y = self.face_layer2(y)
        y = self.face_layer3(y)

        attxy_voice = self.attxy_voice(x.transpose(1, 2))  # 256*80
        attxy_face = torch.sigmoid(self.attxy_face(y.view(y.shape[0], y.shape[1], -1).transpose(1, 2)))  # 256*14*14
        attxy = F.adaptive_avg_pool1d(torch.matmul(attxy_face.transpose(1, 2), attxy_voice).transpose(1, 2), 1)  # 256*1
        attxy_expand = self.attxy_bn(self.attxy_expand(attxy.transpose(1, 2)).transpose(1, 2))  # 256*80
        x = x + attxy_expand
        attyx_voice = torch.sigmoid(self.attyx_voice(x.transpose(1, 2)))  # 256*80
        attyx_face = self.attyx_face(y.view(y.shape[0], y.shape[1], -1).transpose(1, 2))  # 256*14*14
        attyx = F.adaptive_avg_pool1d(torch.matmul(attyx_voice.transpose(1, 2), attyx_face).transpose(1, 2), 1)  # 256*1
        attyx_expand = self.attyx_bn(
            self.attyx_expand(attyx.transpose(1, 2)).transpose(1, 2).view(-1, 256, 14, 14))  # 256*14*14
        y = y + attyx_expand

        x = self.voice_block5(x)
        x = self.voice_bn_before_gru(x)
        x = self.voice_lrelu_keras(x)
        x = x.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
        self.voice_gru.flatten_parameters()
        x, _ = self.voice_gru(x)
        x = x[:, -1, :]
        x = self.voice_fc1_gru(x)

        y = self.face_layer4(y)
        y = self.face_bn2(y)
        y = self.face_dropout(y)
        # endregion # <--- II-Net ---> #

        # region    # <--- Motion ---> #
        z = self.video_conv1(inpt[1])
        z = self.video_bn1(z)
        z = self.video_relu(z)
        z = self.video_maxpool(z)

        z = self.video_layer1(z)
        z = self.video_layer2(z)
        z = self.video_layer3(z)
        z = self.video_layer4(z)
        # endregion # <--- Motion ---> #
        embed_x = self.embedding_aggregator[2](x)
        embed_y = self.embedding_aggregator[0](y)
        embed_z = self.embedding_aggregator[1](z)
        embed = self.embedding_mixer([embed_y, embed_z, embed_x])
        return embed


class MMClassification_XZ1(MMClassification_XY1):
    '''
    follow XY3
    '''

    def __init__(self, args: DictConfig):
        super().__init__(args)
        att_channel = args.cross_attention.middle_channels
        self.attxz_voice = nn.Conv1d(256, att_channel, 1)  # 256 * 80
        self.attxz_video = nn.Conv1d(512, att_channel, 1)  # 512 * 4 * 14 * 14
        self.attxz_expand = nn.Conv1d(1, 256, 1)
        self.attxz_bn = nn.BatchNorm1d(256)
        self.attzx_voice = nn.Conv1d(256, att_channel, 1)
        self.attzx_video = nn.Conv1d(512, att_channel, 1)
        self.attzx_expand = nn.Conv1d(1, 512, 1)
        self.attzx_bn = nn.BatchNorm3d(512)
        try:
            if args.state_dict.split('.')[-1] == "ckpt":
                self.load_state_dict(torch.load(args.state_dict)["state_dict"])
            else:
                self.load_state_dict(torch.load(args.state_dict), strict=False)
            print("XZ1 load ckpt success")
        except Exception:
            print("XZ1 load ckpt failed")

    def forward(self, inpt, **kwargs) -> Any:
        # region    # <--- RawNet ---> #
        nb_samp = inpt[2].shape[0]
        len_seq = inpt[2].shape[1]
        x = self.voice_ln(inpt[2])
        x = x.view(nb_samp, 1, len_seq)
        x = F.max_pool1d(torch.abs(self.voice_first_conv(x)), 3)
        x = self.voice_first_bn(x)
        x = self.voice_lrelu_keras(x)
        x = self.voice_block0(x)
        x = self.voice_block1(x)
        x = self.voice_block2(x)
        x = self.voice_block3(x)
        x = self.voice_block4(x)

        z = self.video_conv1(inpt[1])
        z = self.video_bn1(z)
        z = self.video_relu(z)
        z = self.video_maxpool(z)
        z = self.video_layer1(z)
        z = self.video_layer2(z)

        attxz_voice = self.attxz_voice(x)  # b*256*80 -> b*c*80
        attxz_video = torch.softmax(
            self.attxz_video(torch._adaptive_avg_pool3d(z, (80, 1, 1)).squeeze()), -1)  # b*512*4*14*14 -> b*c*80
        attxz = F.adaptive_avg_pool1d(torch.matmul(attxz_video.transpose(1, 2), attxz_voice).transpose(1, 2),
                                      1)  # matmul(b*80*c, b*c*80)->b*80*80->b*80*1
        attxz_expand = self.attxz_bn(self.attxz_expand(attxz.transpose(1, 2)))  # b*1*80->b*256*80
        x = x + attxz_expand
        attzx_voice = torch.softmax(self.attzx_voice(x), -1)  # b*256*80 -> b*c*80
        attzx_video = self.attzx_video(
            torch._adaptive_avg_pool3d(z, (80, 1, 1)).squeeze())  # b*512*4*14*14 -> b*c*80
        attzx = F.adaptive_avg_pool1d(torch.matmul(attzx_voice.transpose(1, 2), attzx_video).transpose(1, 2),
                                      1)  # matmul(b*80*c, b*c*80)->b*80*80->b*80*1
        attzx_expand = self.attzx_bn(
            torch._adaptive_avg_pool3d(self.attzx_expand(attzx.transpose(1, 2)).unsqueeze(-1).unsqueeze(-1),
                                       (4, 14, 14)))  # b*1*80->b*512*80->b*512*80*1*1
        z = z + attzx_expand

        x = self.voice_block5(x)
        x = self.voice_bn_before_gru(x)
        x = self.voice_lrelu_keras(x)
        x = x.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
        self.voice_gru.flatten_parameters()
        x, _ = self.voice_gru(x)
        x = x[:, -1, :]
        x = self.voice_fc1_gru(x)

        z = self.video_layer3(z)
        z = self.video_layer4(z)

        # endregion # <--- II-Net ---> #
        y = self.face_conv1(inpt[0])
        y = self.face_bn1(y)
        y = self.face_prelu(y)
        y = self.face_layer1(y)
        y = self.face_layer2(y)
        y = self.face_layer3(y)
        y = self.face_layer4(y)
        y = self.face_bn2(y)
        y = self.face_dropout(y)

        # region    # <--- Motion ---> #

        # endregion # <--- Motion ---> #
        embed_x = self.embedding_aggregator[2](x)
        embed_y = self.embedding_aggregator[0](y)
        embed_z = self.embedding_aggregator[1](z)
        embed = self.embedding_mixer([embed_y, embed_z, embed_x])
        return embed


class MMClassification_YZ1(MMClassification_XY1):
    '''
    follow XY3
    '''

    def __init__(self, args: DictConfig):
        super().__init__(args)
        att_channel = args.cross_attention.middle_channels
        self.attyz_face = nn.Conv1d(14 * 14, att_channel, 1)  # 256 * 14 * 14
        self.attyz_video = nn.Conv1d(14 * 14, att_channel, 1)  # 512 * 4 * 14 * 14
        self.attyz_expand = nn.Conv1d(1, 14 * 14, 1)
        self.attyz_bn = nn.BatchNorm2d(256)
        self.attzy_face = nn.Conv1d(14 * 14, att_channel, 1)
        self.attzy_video = nn.Conv1d(14 * 14, att_channel, 1)
        self.attzy_expand = nn.Conv1d(1, 14 * 14, 1)
        self.attzy_bn = nn.BatchNorm3d(512)
        try:
            if args.state_dict.split('.')[-1] == "ckpt":
                self.load_state_dict(torch.load(args.state_dict)["state_dict"])
            else:
                self.load_state_dict(torch.load(args.state_dict), strict=False)
            print("YZ1 load ckpt success")
        except Exception:
            print("YZ1 load ckpt failed")

    def forward(self, inpt, **kwargs) -> Any:
        # region    # <--- RawNet ---> #
        nb_samp = inpt[2].shape[0]
        len_seq = inpt[2].shape[1]
        x = self.voice_ln(inpt[2])
        x = x.view(nb_samp, 1, len_seq)
        x = F.max_pool1d(torch.abs(self.voice_first_conv(x)), 3)
        x = self.voice_first_bn(x)
        x = self.voice_lrelu_keras(x)
        x = self.voice_block0(x)
        x = self.voice_block1(x)
        x = self.voice_block2(x)
        x = self.voice_block3(x)
        x = self.voice_block4(x)
        x = self.voice_block5(x)
        x = self.voice_bn_before_gru(x)
        x = self.voice_lrelu_keras(x)
        x = x.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
        self.voice_gru.flatten_parameters()
        x, _ = self.voice_gru(x)
        x = x[:, -1, :]
        x = self.voice_fc1_gru(x)
        # endregion # <--- II-Net ---> #

        y = self.face_conv1(inpt[0])
        y = self.face_bn1(y)
        y = self.face_prelu(y)
        y = self.face_layer1(y)
        y = self.face_layer2(y)
        y = self.face_layer3(y)

        z = self.video_conv1(inpt[1])
        z = self.video_bn1(z)
        z = self.video_relu(z)
        z = self.video_maxpool(z)
        z = self.video_layer1(z)
        z = self.video_layer2(z)

        attyz_face = self.attyz_face(y.view(y.shape[0], y.shape[1], -1).transpose(1, 2))  # b*256*14*14 -> b*c*256
        attyz_video = torch.softmax(self.attyz_video(
            torch._adaptive_avg_pool3d(z, (1, 14, 14)).squeeze().view(z.shape[0], z.shape[1], -1).transpose(1, 2)),
            -1)  # b*512*4*14*14 -> b*512*14*14 -> b*512*196 -> b*c*512
        attyz = F.adaptive_avg_pool1d(torch.matmul(attyz_video.transpose(1, 2), attyz_face).transpose(1, 2),
                                      1)  # matmul(b*512*c, b*c*256)->b*512*256->b*256*1
        attyz_expand = self.attyz_bn(self.attyz_expand(attyz.transpose(1, 2)).transpose(1, 2).view(-1, 256, 14,
                                                                                                   14))  # b*1*256->b*196*256->b*256*14*14
        y = y + attyz_expand
        attzy_face = torch.softmax(self.attzy_face(y.view(y.shape[0], y.shape[1], -1).transpose(1, 2)),
                                   -1)  # b*256*14*14 -> b*c*256
        attzy_video = self.attzy_video(
            torch._adaptive_avg_pool3d(z, (1, 14, 14)).squeeze().view(z.shape[0], z.shape[1], -1).transpose(1,
                                                                                                            2))  # b*512*4*14*14 -> b*512*14*14 -> b*512*196 -> b*c*512
        attzy = F.adaptive_avg_pool1d(torch.matmul(attzy_face.transpose(1, 2), attzy_video).transpose(1, 2),
                                      1)  # matmul(b*256*c, b*c*512)->b*256*512->b*512*1
        attzy_expand = self.attzy_bn(
            torch._adaptive_avg_pool3d(
                self.attzy_expand(attzy.transpose(1, 2)).transpose(1, 2).view(z.shape[0], z.shape[1], 14, 14).unsqueeze(
                    2),
                (4, 14, 14)))  # b*1*512->b*196*512->b*512*196->b*512*14*14
        z = z + attzy_expand

        y = self.face_layer4(y)
        y = self.face_bn2(y)
        y = self.face_dropout(y)

        z = self.video_layer3(z)
        z = self.video_layer4(z)

        # region    # <--- Motion ---> #

        # endregion # <--- Motion ---> #
        embed_x = self.embedding_aggregator[2](x)
        embed_y = self.embedding_aggregator[0](y)
        embed_z = self.embedding_aggregator[1](z)
        embed = self.embedding_mixer([embed_y, embed_z, embed_x])
        return embed


# endregion # <--- LCMPA-Single ---> #

# region    # <--- LCMPA-Combinations ---> #

class MMClassification_XY_XZ1(MMClassification_XY1):
    '''
    follow XY3
    '''

    def __init__(self, args: DictConfig):
        super().__init__(args)
        att_channel = args.cross_attention.middle_channels

        self.attxy_voice = nn.Conv1d(80, att_channel, 1)
        self.attxy_face = nn.Conv1d(14 * 14, att_channel, 1)
        self.attxy_expand = nn.Conv1d(1, 80, 1)
        self.attxy_bn = nn.BatchNorm1d(256)
        self.attyx_voice = nn.Conv1d(80, att_channel, 1)
        self.attyx_face = nn.Conv1d(14 * 14, att_channel, 1)
        self.attyx_expand = nn.Conv1d(1, 14 * 14, 1)
        self.attyx_bn = nn.BatchNorm2d(256)

        self.attxz_voice = nn.Conv1d(256, att_channel, 1)  # 256 * 80
        self.attxz_video = nn.Conv1d(512, att_channel, 1)  # 512 * 4 * 14 * 14
        self.attxz_expand = nn.Conv1d(1, 256, 1)
        self.attxz_bn = nn.BatchNorm1d(256)
        self.attzx_voice = nn.Conv1d(256, att_channel, 1)
        self.attzx_video = nn.Conv1d(512, att_channel, 1)
        self.attzx_expand = nn.Conv1d(1, 512, 1)
        self.attzx_bn = nn.BatchNorm3d(512)

        try:
            if args.state_dict.split('.')[-1] == "ckpt":
                self.load_state_dict(torch.load(args.state_dict)["state_dict"])
            else:
                self.load_state_dict(torch.load(args.state_dict), strict=False)
            print("XY_XZ1 load ckpt success")
        except Exception:
            print("XY_XZ1 load ckpt failed")

    def forward(self, inpt, **kwargs) -> Any:
        # region    # <--- before ---> #
        nb_samp = inpt[2].shape[0]
        len_seq = inpt[2].shape[1]
        x = self.voice_ln(inpt[2])
        x = x.view(nb_samp, 1, len_seq)
        x = F.max_pool1d(torch.abs(self.voice_first_conv(x)), 3)
        x = self.voice_first_bn(x)
        x = self.voice_lrelu_keras(x)
        x = self.voice_block0(x)
        x = self.voice_block1(x)
        x = self.voice_block2(x)
        x = self.voice_block3(x)
        x = self.voice_block4(x)

        y = self.face_conv1(inpt[0])
        y = self.face_bn1(y)
        y = self.face_prelu(y)
        y = self.face_layer1(y)
        y = self.face_layer2(y)
        y = self.face_layer3(y)

        z = self.video_conv1(inpt[1])
        z = self.video_bn1(z)
        z = self.video_relu(z)
        z = self.video_maxpool(z)
        z = self.video_layer1(z)
        z = self.video_layer2(z)
        # endregion # <--- before ---> #

        attxy_voice = self.attxy_voice(x.transpose(1, 2))  # 256*80
        attxy_face = torch.softmax(self.attxy_face(y.view(y.shape[0], y.shape[1], -1).transpose(1, 2)), -1)  # 256*14*14
        attxy = F.adaptive_avg_pool1d(torch.matmul(attxy_face.transpose(1, 2), attxy_voice).transpose(1, 2), 1)  # 256*1
        attxy_expand = self.attxy_bn(self.attxy_expand(attxy.transpose(1, 2)).transpose(1, 2))  # 256*80
        attyx_voice = torch.softmax(self.attyx_voice(x.transpose(1, 2)), -1)  # 256*80
        attyx_face = self.attyx_face(y.view(y.shape[0], y.shape[1], -1).transpose(1, 2))  # 256*14*14
        attyx = F.adaptive_avg_pool1d(torch.matmul(attyx_voice.transpose(1, 2), attyx_face).transpose(1, 2), 1)  # 256*1
        attyx_expand = self.attyx_bn(
            self.attyx_expand(attyx.transpose(1, 2)).transpose(1, 2).view(-1, 256, 14, 14))  # 256*14*14

        attxz_voice = self.attxz_voice(x)  # b*256*80 -> b*c*80
        attxz_video = torch.softmax(self.attxz_video(F.adaptive_avg_pool3d(z, (80, 1, 1)).squeeze()),
                                    -1)  # b*512*4*14*14 -> b*c*80
        attxz = F.adaptive_avg_pool1d(torch.matmul(attxz_video.transpose(1, 2), attxz_voice).transpose(1, 2),
                                      1)  # matmul(b*80*c, b*c*80)->b*80*80->b*80*1
        attxz_expand = self.attxz_bn(self.attxz_expand(attxz.transpose(1, 2)))  # b*1*80->b*256*80
        attzx_voice = torch.softmax(self.attzx_voice(x), -1)  # b*256*80 -> b*c*80
        attzx_video = self.attzx_video(F.adaptive_avg_pool3d(z, (80, 1, 1)).squeeze())  # b*512*4*14*14 -> b*c*80
        attzx = F.adaptive_avg_pool1d(torch.matmul(attzx_voice.transpose(1, 2), attzx_video).transpose(1, 2),
                                      1)  # matmul(b*80*c, b*c*80)->b*80*80->b*80*1
        attzx_expand = self.attzx_bn(
            F.adaptive_avg_pool3d(self.attzx_expand(attzx.transpose(1, 2)).unsqueeze(-1).unsqueeze(-1),
                                  (4, 14, 14)))  # b*1*80->b*512*80->b*512*80*1*1

        x = x + attxy_expand + attxz_expand
        y = y + attyx_expand
        z = z + attzx_expand

        # region    # <--- after ---> #

        x = self.voice_block5(x)
        x = self.voice_bn_before_gru(x)
        x = self.voice_lrelu_keras(x)
        x = x.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
        self.voice_gru.flatten_parameters()
        x, _ = self.voice_gru(x)
        x = x[:, -1, :]
        x = self.voice_fc1_gru(x)

        y = self.face_layer4(y)
        y = self.face_bn2(y)
        y = self.face_dropout(y)

        z = self.video_layer3(z)
        z = self.video_layer4(z)

        # endregion # <--- after ---> #

        embed_x = self.embedding_aggregator[2](x)
        embed_y = self.embedding_aggregator[0](y)
        embed_z = self.embedding_aggregator[1](z)
        embed = self.embedding_mixer([embed_y, embed_z, embed_x])
        return embed


class MMClassification_XY_YZ1(MMClassification_XY1):
    '''
    follow XY3
    '''

    def __init__(self, args: DictConfig):
        super().__init__(args)
        att_channel = args.cross_attention.middle_channels

        self.attxy_voice = nn.Conv1d(80, att_channel, 1)
        self.attxy_face = nn.Conv1d(14 * 14, att_channel, 1)
        self.attxy_expand = nn.Conv1d(1, 80, 1)
        self.attxy_bn = nn.BatchNorm1d(256)
        self.attyx_voice = nn.Conv1d(80, att_channel, 1)
        self.attyx_face = nn.Conv1d(14 * 14, att_channel, 1)
        self.attyx_expand = nn.Conv1d(1, 14 * 14, 1)
        self.attyx_bn = nn.BatchNorm2d(256)

        self.attyz_face = nn.Conv1d(14 * 14, att_channel, 1)  # 256 * 14 * 14
        self.attyz_video = nn.Conv1d(14 * 14, att_channel, 1)  # 512 * 4 * 14 * 14
        self.attyz_expand = nn.Conv1d(1, 14 * 14, 1)
        self.attyz_bn = nn.BatchNorm2d(256)
        self.attzy_face = nn.Conv1d(14 * 14, att_channel, 1)
        self.attzy_video = nn.Conv1d(14 * 14, att_channel, 1)
        self.attzy_expand = nn.Conv1d(1, 14 * 14, 1)
        self.attzy_bn = nn.BatchNorm3d(512)

        try:
            if args.state_dict.split('.')[-1] == "ckpt":
                self.load_state_dict(torch.load(args.state_dict)["state_dict"])
            else:
                self.load_state_dict(torch.load(args.state_dict), strict=False)
            print("XY_YZ1 load ckpt success")
        except Exception:
            print("XY_YZ1 load ckpt failed")

    def forward(self, inpt, **kwargs) -> Any:
        # region    # <--- before ---> #
        nb_samp = inpt[2].shape[0]
        len_seq = inpt[2].shape[1]
        x = self.voice_ln(inpt[2])
        x = x.view(nb_samp, 1, len_seq)
        x = F.max_pool1d(torch.abs(self.voice_first_conv(x)), 3)
        x = self.voice_first_bn(x)
        x = self.voice_lrelu_keras(x)
        x = self.voice_block0(x)
        x = self.voice_block1(x)
        x = self.voice_block2(x)
        x = self.voice_block3(x)
        x = self.voice_block4(x)

        y = self.face_conv1(inpt[0])
        y = self.face_bn1(y)
        y = self.face_prelu(y)
        y = self.face_layer1(y)
        y = self.face_layer2(y)
        y = self.face_layer3(y)

        z = self.video_conv1(inpt[1])
        z = self.video_bn1(z)
        z = self.video_relu(z)
        z = self.video_maxpool(z)
        z = self.video_layer1(z)
        z = self.video_layer2(z)
        # endregion # <--- before ---> #

        attxy_voice = self.attxy_voice(x.transpose(1, 2))  # 256*80
        attxy_face = torch.softmax(self.attxy_face(y.view(y.shape[0], y.shape[1], -1).transpose(1, 2)), -1)  # 256*14*14
        attxy = F.adaptive_avg_pool1d(torch.matmul(attxy_face.transpose(1, 2), attxy_voice).transpose(1, 2), 1)  # 256*1
        attxy_expand = self.attxy_bn(self.attxy_expand(attxy.transpose(1, 2)).transpose(1, 2))  # 256*80
        attyx_voice = torch.softmax(self.attyx_voice(x.transpose(1, 2)), -1)  # 256*80
        attyx_face = self.attyx_face(y.view(y.shape[0], y.shape[1], -1).transpose(1, 2))  # 256*14*14
        attyx = F.adaptive_avg_pool1d(torch.matmul(attyx_voice.transpose(1, 2), attyx_face).transpose(1, 2), 1)  # 256*1
        attyx_expand = self.attyx_bn(
            self.attyx_expand(attyx.transpose(1, 2)).transpose(1, 2).view(-1, 256, 14, 14))  # 256*14*14

        attyz_face = self.attyz_face(y.view(y.shape[0], y.shape[1], -1).transpose(1, 2))  # b*256*14*14 -> b*c*256
        attyz_video = torch.softmax(self.attyz_video(
            F.adaptive_avg_pool3d(z, (1, 14, 14)).squeeze().view(z.shape[0], z.shape[1], -1).transpose(1, 2)),
            -1)  # b*512*4*14*14 -> b*512*14*14 -> b*512*196 -> b*c*512
        attyz = F.adaptive_avg_pool1d(torch.matmul(attyz_video.transpose(1, 2), attyz_face).transpose(1, 2),
                                      1)  # matmul(b*512*c, b*c*256)->b*512*256->b*256*1
        attyz_expand = self.attyz_bn(self.attyz_expand(attyz.transpose(1, 2)).transpose(1, 2).view(-1, 256, 14,
                                                                                                   14))  # b*1*256->b*196*256->b*256*14*14
        attzy_face = torch.softmax(self.attzy_face(y.view(y.shape[0], y.shape[1], -1).transpose(1, 2)),
                                   -1)  # b*256*14*14 -> b*c*256
        attzy_video = self.attzy_video(
            F.adaptive_avg_pool3d(z, (1, 14, 14)).squeeze().view(z.shape[0], z.shape[1], -1).transpose(1,
                                                                                                       2))  # b*512*4*14*14 -> b*512*14*14 -> b*512*196 -> b*c*512
        attzy = F.adaptive_avg_pool1d(torch.matmul(attzy_face.transpose(1, 2), attzy_video).transpose(1, 2),
                                      1)  # matmul(b*256*c, b*c*512)->b*256*512->b*512*1
        attzy_expand = self.attzy_bn(
            F.adaptive_avg_pool3d(
                self.attzy_expand(attzy.transpose(1, 2)).transpose(1, 2).view(z.shape[0], z.shape[1], 14, 14).unsqueeze(
                    2),
                (4, 14, 14)))  # b*1*512->b*196*512->b*512*196->b*512*14*14

        x = x + attxy_expand
        y = y + attyx_expand + attyz_expand
        z = z + attzy_expand

        # region    # <--- after ---> #

        x = self.voice_block5(x)
        x = self.voice_bn_before_gru(x)
        x = self.voice_lrelu_keras(x)
        x = x.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
        self.voice_gru.flatten_parameters()
        x, _ = self.voice_gru(x)
        x = x[:, -1, :]
        x = self.voice_fc1_gru(x)

        y = self.face_layer4(y)
        y = self.face_bn2(y)
        y = self.face_dropout(y)

        z = self.video_layer3(z)
        z = self.video_layer4(z)

        # endregion # <--- after ---> #

        embed_x = self.embedding_aggregator[2](x)
        embed_y = self.embedding_aggregator[0](y)
        embed_z = self.embedding_aggregator[1](z)
        embed = self.embedding_mixer([embed_y, embed_z, embed_x])
        return embed


class MMClassification_XZ_YZ1(MMClassification_XY1):
    '''
    follow XY3
    '''

    def __init__(self, args: DictConfig):
        super().__init__(args)
        att_channel = args.cross_attention.middle_channels

        self.attxz_voice = nn.Conv1d(256, att_channel, 1)  # 256 * 80
        self.attxz_video = nn.Conv1d(512, att_channel, 1)  # 512 * 4 * 14 * 14
        self.attxz_expand = nn.Conv1d(1, 256, 1)
        self.attxz_bn = nn.BatchNorm1d(256)
        self.attzx_voice = nn.Conv1d(256, att_channel, 1)
        self.attzx_video = nn.Conv1d(512, att_channel, 1)
        self.attzx_expand = nn.Conv1d(1, 512, 1)
        self.attzx_bn = nn.BatchNorm3d(512)

        self.attyz_face = nn.Conv1d(14 * 14, att_channel, 1)  # 256 * 14 * 14
        self.attyz_video = nn.Conv1d(14 * 14, att_channel, 1)  # 512 * 4 * 14 * 14
        self.attyz_expand = nn.Conv1d(1, 14 * 14, 1)
        self.attyz_bn = nn.BatchNorm2d(256)
        self.attzy_face = nn.Conv1d(14 * 14, att_channel, 1)
        self.attzy_video = nn.Conv1d(14 * 14, att_channel, 1)
        self.attzy_expand = nn.Conv1d(1, 14 * 14, 1)
        self.attzy_bn = nn.BatchNorm3d(512)

        try:
            if args.state_dict.split('.')[-1] == "ckpt":
                self.load_state_dict(torch.load(args.state_dict)["state_dict"])
            else:
                self.load_state_dict(torch.load(args.state_dict), strict=False)
            print("XZ_YZ1 load ckpt success")
        except Exception:
            print("XZ_YZ1 load ckpt failed")

    def forward(self, inpt, **kwargs) -> Any:
        # region    # <--- before ---> #
        nb_samp = inpt[2].shape[0]
        len_seq = inpt[2].shape[1]
        x = self.voice_ln(inpt[2])
        x = x.view(nb_samp, 1, len_seq)
        x = F.max_pool1d(torch.abs(self.voice_first_conv(x)), 3)
        x = self.voice_first_bn(x)
        x = self.voice_lrelu_keras(x)
        x = self.voice_block0(x)
        x = self.voice_block1(x)
        x = self.voice_block2(x)
        x = self.voice_block3(x)
        x = self.voice_block4(x)

        y = self.face_conv1(inpt[0])
        y = self.face_bn1(y)
        y = self.face_prelu(y)
        y = self.face_layer1(y)
        y = self.face_layer2(y)
        y = self.face_layer3(y)

        z = self.video_conv1(inpt[1])
        z = self.video_bn1(z)
        z = self.video_relu(z)
        z = self.video_maxpool(z)
        z = self.video_layer1(z)
        z = self.video_layer2(z)
        # endregion # <--- before ---> #

        attxz_voice = self.attxz_voice(x)  # b*256*80 -> b*c*80
        attxz_video = torch.softmax(self.attxz_video(F.adaptive_avg_pool3d(z, (80, 1, 1)).squeeze()),
                                    -1)  # b*512*4*14*14 -> b*c*80
        attxz = F.adaptive_avg_pool1d(torch.matmul(attxz_video.transpose(1, 2), attxz_voice).transpose(1, 2),
                                      1)  # matmul(b*80*c, b*c*80)->b*80*80->b*80*1
        attxz_expand = self.attxz_bn(self.attxz_expand(attxz.transpose(1, 2)))  # b*1*80->b*256*80
        attzx_voice = torch.softmax(self.attzx_voice(x), -1)  # b*256*80 -> b*c*80
        attzx_video = self.attzx_video(F.adaptive_avg_pool3d(z, (80, 1, 1)).squeeze())  # b*512*4*14*14 -> b*c*80
        attzx = F.adaptive_avg_pool1d(torch.matmul(attzx_voice.transpose(1, 2), attzx_video).transpose(1, 2),
                                      1)  # matmul(b*80*c, b*c*80)->b*80*80->b*80*1
        attzx_expand = self.attzx_bn(
            F.adaptive_avg_pool3d(self.attzx_expand(attzx.transpose(1, 2)).unsqueeze(-1).unsqueeze(-1),
                                  (4, 14, 14)))  # b*1*80->b*512*80->b*512*80*1*1

        attyz_face = self.attyz_face(y.view(y.shape[0], y.shape[1], -1).transpose(1, 2))  # b*256*14*14 -> b*c*256
        attyz_video = torch.softmax(self.attyz_video(
            F.adaptive_avg_pool3d(z, (1, 14, 14)).squeeze().view(z.shape[0], z.shape[1], -1).transpose(1, 2)),
            -1)  # b*512*4*14*14 -> b*512*14*14 -> b*512*196 -> b*c*512
        attyz = F.adaptive_avg_pool1d(torch.matmul(attyz_video.transpose(1, 2), attyz_face).transpose(1, 2),
                                      1)  # matmul(b*512*c, b*c*256)->b*512*256->b*256*1
        attyz_expand = self.attyz_bn(self.attyz_expand(attyz.transpose(1, 2)).transpose(1, 2).view(-1, 256, 14,
                                                                                                   14))  # b*1*256->b*196*256->b*256*14*14
        attzy_face = torch.softmax(self.attzy_face(y.view(y.shape[0], y.shape[1], -1).transpose(1, 2)),
                                   -1)  # b*256*14*14 -> b*c*256
        attzy_video = self.attzy_video(
            F.adaptive_avg_pool3d(z, (1, 14, 14)).squeeze().view(z.shape[0], z.shape[1], -1).transpose(1,
                                                                                                       2))  # b*512*4*14*14 -> b*512*14*14 -> b*512*196 -> b*c*512
        attzy = F.adaptive_avg_pool1d(torch.matmul(attzy_face.transpose(1, 2), attzy_video).transpose(1, 2),
                                      1)  # matmul(b*256*c, b*c*512)->b*256*512->b*512*1
        attzy_expand = self.attzy_bn(
            F.adaptive_avg_pool3d(
                self.attzy_expand(attzy.transpose(1, 2)).transpose(1, 2).view(z.shape[0], z.shape[1], 14, 14).unsqueeze(
                    2),
                (4, 14, 14)))  # b*1*512->b*196*512->b*512*196->b*512*14*14

        x = x + attxz_expand
        y = y + attyz_expand
        z = z + attzx_expand + attzy_expand

        # region    # <--- after ---> #

        x = self.voice_block5(x)
        x = self.voice_bn_before_gru(x)
        x = self.voice_lrelu_keras(x)
        x = x.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
        self.voice_gru.flatten_parameters()
        x, _ = self.voice_gru(x)
        x = x[:, -1, :]
        x = self.voice_fc1_gru(x)

        y = self.face_layer4(y)
        y = self.face_bn2(y)
        y = self.face_dropout(y)

        z = self.video_layer3(z)
        z = self.video_layer4(z)

        # endregion # <--- after ---> #

        embed_x = self.embedding_aggregator[2](x)
        embed_y = self.embedding_aggregator[0](y)
        embed_z = self.embedding_aggregator[1](z)
        embed = self.embedding_mixer([embed_y, embed_z, embed_x])
        return embed


class MMClassification_XYZ1(MMClassification_XY1):
    '''
    follow XY3
    '''

    def __init__(self, args: DictConfig):
        super().__init__(args)
        att_channel = args.cross_attention.middle_channels

        self.attxy_voice = nn.Conv1d(80, att_channel, 1)
        self.attxy_face = nn.Conv1d(14 * 14, att_channel, 1)
        self.attxy_expand = nn.Conv1d(1, 80, 1)
        self.attxy_bn = nn.BatchNorm1d(256)
        self.attyx_voice = nn.Conv1d(80, att_channel, 1)
        self.attyx_face = nn.Conv1d(14 * 14, att_channel, 1)
        self.attyx_expand = nn.Conv1d(1, 14 * 14, 1)
        self.attyx_bn = nn.BatchNorm2d(256)

        self.attyz_face = nn.Conv1d(14 * 14, att_channel, 1)  # 256 * 14 * 14
        self.attyz_video = nn.Conv1d(14 * 14, att_channel, 1)  # 512 * 4 * 14 * 14
        self.attyz_expand = nn.Conv1d(1, 14 * 14, 1)
        self.attyz_bn = nn.BatchNorm2d(256)
        self.attzy_face = nn.Conv1d(14 * 14, att_channel, 1)
        self.attzy_video = nn.Conv1d(14 * 14, att_channel, 1)
        self.attzy_expand = nn.Conv1d(1, 14 * 14, 1)
        self.attzy_bn = nn.BatchNorm3d(512)

        try:
            if args.state_dict.split('.')[-1] == "ckpt":
                self.load_state_dict(torch.load(args.state_dict)["state_dict"])
            else:
                self.load_state_dict(torch.load(args.state_dict), strict=False)
            print("XYZ1 load ckpt success")
        except Exception:
            print("XYZ1 load ckpt failed")

    def forward(self, inpt, **kwargs) -> Any:
        # region    # <--- before ---> #
        nb_samp = inpt[2].shape[0]
        len_seq = inpt[2].shape[1]
        x = self.voice_ln(inpt[2])
        x = x.view(nb_samp, 1, len_seq)
        x = F.max_pool1d(torch.abs(self.voice_first_conv(x)), 3)
        x = self.voice_first_bn(x)
        x = self.voice_lrelu_keras(x)
        x = self.voice_block0(x)
        x = self.voice_block1(x)
        x = self.voice_block2(x)
        x = self.voice_block3(x)
        x = self.voice_block4(x)

        y = self.face_conv1(inpt[0])
        y = self.face_bn1(y)
        y = self.face_prelu(y)
        y = self.face_layer1(y)
        y = self.face_layer2(y)
        y = self.face_layer3(y)

        z = self.video_conv1(inpt[1])
        z = self.video_bn1(z)
        z = self.video_relu(z)
        z = self.video_maxpool(z)
        z = self.video_layer1(z)
        z = self.video_layer2(z)
        # endregion # <--- before ---> #

        attxy_voice = self.attxy_voice(x.transpose(1, 2))  # 256*80
        attxy_face = torch.softmax(self.attxy_face(y.view(y.shape[0], y.shape[1], -1).transpose(1, 2)), -1)  # 256*14*14
        attxy = F.adaptive_avg_pool1d(torch.matmul(attxy_face.transpose(1, 2), attxy_voice).transpose(1, 2), 1)  # 256*1
        attxy_expand = self.attxy_bn(self.attxy_expand(attxy.transpose(1, 2)).transpose(1, 2))  # 256*80
        attyx_voice = torch.softmax(self.attyx_voice(x.transpose(1, 2)), -1)  # 256*80
        attyx_face = self.attyx_face(y.view(y.shape[0], y.shape[1], -1).transpose(1, 2))  # 256*14*14
        attyx = F.adaptive_avg_pool1d(torch.matmul(attyx_voice.transpose(1, 2), attyx_face).transpose(1, 2), 1)  # 256*1
        attyx_expand = self.attyx_bn(
            self.attyx_expand(attyx.transpose(1, 2)).transpose(1, 2).view(-1, 256, 14, 14))  # 256*14*14

        attxz_voice = self.attxz_voice(x)  # b*256*80 -> b*c*80
        attxz_video = torch.softmax(self.attxz_video(F.adaptive_avg_pool3d(z, (80, 1, 1)).squeeze()),
                                    -1)  # b*512*4*14*14 -> b*c*80
        attxz = F.adaptive_avg_pool1d(torch.matmul(attxz_video.transpose(1, 2), attxz_voice).transpose(1, 2),
                                      1)  # matmul(b*80*c, b*c*80)->b*80*80->b*80*1
        attxz_expand = self.attxz_bn(self.attxz_expand(attxz.transpose(1, 2)))  # b*1*80->b*256*80
        attzx_voice = torch.softmax(self.attzx_voice(x), -1)  # b*256*80 -> b*c*80
        attzx_video = self.attzx_video(F.adaptive_avg_pool3d(z, (80, 1, 1)).squeeze())  # b*512*4*14*14 -> b*c*80
        attzx = F.adaptive_avg_pool1d(torch.matmul(attzx_voice.transpose(1, 2), attzx_video).transpose(1, 2),
                                      1)  # matmul(b*80*c, b*c*80)->b*80*80->b*80*1
        attzx_expand = self.attzx_bn(
            F.adaptive_avg_pool3d(self.attzx_expand(attzx.transpose(1, 2)).unsqueeze(-1).unsqueeze(-1),
                                  (4, 14, 14)))  # b*1*80->b*512*80->b*512*80*1*1

        attyz_face = self.attyz_face(y.view(y.shape[0], y.shape[1], -1).transpose(1, 2))  # b*256*14*14 -> b*c*256
        attyz_video = torch.softmax(self.attyz_video(
            F.adaptive_avg_pool3d(z, (1, 14, 14)).squeeze().view(z.shape[0], z.shape[1], -1).transpose(1, 2)),
            -1)  # b*512*4*14*14 -> b*512*14*14 -> b*512*196 -> b*c*512
        attyz = F.adaptive_avg_pool1d(torch.matmul(attyz_video.transpose(1, 2), attyz_face).transpose(1, 2),
                                      1)  # matmul(b*512*c, b*c*256)->b*512*256->b*256*1
        attyz_expand = self.attyz_bn(self.attyz_expand(attyz.transpose(1, 2)).transpose(1, 2).view(-1, 256, 14,
                                                                                                   14))  # b*1*256->b*196*256->b*256*14*14
        attzy_face = torch.softmax(self.attzy_face(y.view(y.shape[0], y.shape[1], -1).transpose(1, 2)),
                                   -1)  # b*256*14*14 -> b*c*256
        attzy_video = self.attzy_video(
            F.adaptive_avg_pool3d(z, (1, 14, 14)).squeeze().view(z.shape[0], z.shape[1], -1).transpose(1,
                                                                                                       2))  # b*512*4*14*14 -> b*512*14*14 -> b*512*196 -> b*c*512
        attzy = F.adaptive_avg_pool1d(torch.matmul(attzy_face.transpose(1, 2), attzy_video).transpose(1, 2),
                                      1)  # matmul(b*256*c, b*c*512)->b*256*512->b*512*1
        attzy_expand = self.attzy_bn(
            F.adaptive_avg_pool3d(
                self.attzy_expand(attzy.transpose(1, 2)).transpose(1, 2).view(z.shape[0], z.shape[1], 14, 14).unsqueeze(
                    2),
                (4, 14, 14)))  # b*1*512->b*196*512->b*512*196->b*512*14*14

        x = x + attxy_expand + attxz_expand
        y = y + attyx_expand + attyz_expand
        z = z + attzx_expand + attzy_expand

        # region    # <--- after ---> #

        x = self.voice_block5(x)
        x = self.voice_bn_before_gru(x)
        x = self.voice_lrelu_keras(x)
        x = x.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
        self.voice_gru.flatten_parameters()
        x, _ = self.voice_gru(x)
        x = x[:, -1, :]
        x = self.voice_fc1_gru(x)

        y = self.face_layer4(y)
        y = self.face_bn2(y)
        y = self.face_dropout(y)

        z = self.video_layer3(z)
        z = self.video_layer4(z)

        # endregion # <--- after ---> #

        embed_x = self.embedding_aggregator[2](x)
        embed_y = self.embedding_aggregator[0](y)
        embed_z = self.embedding_aggregator[1](z)
        embed = self.embedding_mixer([embed_y, embed_z, embed_x])
        return embed


# endregion # <--- LCMPA-Combination ---> #

# region    # <--- Compare ---> #

class MFFN_BASE(pl.LightningModule):
    def __init__(self, args: DictConfig):
        super().__init__()
        self.example_input_array = [[torch.rand((2, 3, args.image_size, args.image_size)).to(args.device),
                                     torch.rand((2, 3, 16, args.image_size, args.image_size)).to(args.device),
                                     torch.rand((2, args.nb_samp)).to(args.device)]]
        self.args = args
        self.batch_size = self.args.batch_size
        self.learning_rate = self.args.learning_rate

        # loading Feature Extractors
        # region    # <--- RawNet ---> #
        self.voice_ln = LayerNorm(59049)
        self.voice_first_conv = SincConv_fast(in_channels=1,
                                              out_channels=128,
                                              kernel_size=251)

        self.voice_first_bn = nn.BatchNorm1d(num_features=128)
        self.voice_lrelu = nn.LeakyReLU()
        self.voice_lrelu_keras = nn.LeakyReLU(negative_slope=0.3)

        self.voice_block0 = nn.Sequential(Residual_block_wFRM(nb_filts=[128, 128], first=True))
        self.voice_block1 = nn.Sequential(Residual_block_wFRM(nb_filts=[128, 128]))
        self.voice_block2 = nn.Sequential(Residual_block_wFRM(nb_filts=[128, 256]))
        self.voice_block3 = nn.Sequential(Residual_block_NL(nb_filts=[256, 256, 256, 64]))
        self.voice_block4 = nn.Sequential(Residual_block_NL(nb_filts=[256, 256, 256, 64]))
        self.voice_block5 = nn.Sequential(Residual_block_NL(nb_filts=[256, 256, 256, 64]))

        self.voice_bn_before_gru = nn.BatchNorm1d(num_features=256)
        self.voice_gru = nn.GRU(input_size=256,
                                hidden_size=1024,
                                num_layers=1,
                                bidirectional=False,
                                batch_first=True)
        self.voice_fc1_gru = nn.Linear(in_features=1024,
                                       out_features=1024)
        self.voice_sig = nn.Sigmoid()
        # endregion # <--- RawNet ---> #

        # region    # <--- II-Net ---> #

        self.face_in_channels = 64
        self.face_conv1 = nn.Conv2d(3, 64,
                                    kernel_size=(3, 3), stride=(1, 1),
                                    padding=(1, 1), bias=False)
        self.face_bn1 = nn.BatchNorm2d(64)
        self.face_prelu = nn.PReLU()
        self.face_layer1 = self._make_layer_face(PolyBlock, 64, 3, stride=2)
        self.face_layer2 = self._make_layer_face(PolyBlock, 128, 4, stride=2)
        self.face_layer3 = self._make_layer_face(PolyBlock, 256, 6, stride=2)
        self.face_layer4 = self._make_layer_face(PolyBlock, 512, 3, stride=2)
        self.face_bn2 = nn.BatchNorm2d(512)
        self.face_dropout = nn.Dropout(0)

        # endregion # <--- II-Net ---> #

        # region    # <--- Motion ---> #

        self.video_in_planes = 128
        self.video_cardinality = 32
        self.video_STDA_input_shape = [1024, 2, 7, 7]
        self.video_conv1 = nn.Conv3d(3,
                                     self.video_in_planes,
                                     kernel_size=(7, 7, 7),
                                     stride=(1, 2, 2),
                                     padding=(7 // 2, 3, 3),
                                     bias=False)
        self.video_bn1 = nn.BatchNorm3d(self.video_in_planes)
        self.video_relu = nn.ReLU(inplace=True)
        self.video_maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.video_layer1 = self._make_layer_video(ResNeXtBottleneck3D, 3, 128,
                                                   "B")
        self.video_layer2 = self._make_layer_video(ResNeXtBottleneck3D,
                                                   4,
                                                   256,
                                                   "B",
                                                   stride=2)
        self.video_layer3 = self._make_layer_video(ResNeXtBottleneck3D,
                                                   23,
                                                   512,
                                                   "B",
                                                   stride=2)
        self.video_layer4 = self._make_layer_video(STDABottleneck,
                                                   3,
                                                   1024,
                                                   "B",
                                                   stride=2)

        # endregion # <--- Motion ---> #

        for layer in self.named_modules():
            if "attxy" not in layer[0] and "attyx" not in layer[0] and \
                    "attxz" not in layer[0] and "attzx" not in layer[0] and \
                    "attyz" not in layer[0] and "attyz" not in layer[0]:
                layer[1].eval().requires_grad_(False)
        for layer in self.named_modules():
            if "all" in args.grad:
                layer[1].train().requires_grad_()
            if layer[0] in args.grad:
                layer[1].train().requires_grad_()
            elif "layer" in layer[0] or "block" in layer[0]:
                if ".".join(layer[0].split(".")[:-1]) in args.grad:
                    layer[1].train().requires_grad_()
                if ".".join(layer[0].split(".")[:-2]) in args.grad:
                    layer[1].train().requires_grad_()

        # loading Embedding Aggregators
        if isinstance(args.embedding_aggregator.__target__, ListConfig):
            self.embedding_aggregator = []
            for i in range(len(args.embedding_aggregator.__target__)):
                __E_pkg__ = ".".join(args.embedding_aggregator.__target__[i].split(".")[:-1])
                __E_name__ = args.embedding_aggregator.__target__[i].split(".")[-1]
                __E__ = importlib.import_module(__E_pkg__).__getattribute__(__E_name__)
                if issubclass(__E__, pl.LightningModule):
                    embedding_aggregator = __E__.load_from_checkpoint(args.embedding_aggregator.args[i].checkpoint,
                                                                      args=OmegaConf.load(
                                                                          args.embedding_aggregator.args[
                                                                              i].cfg).experiments.model)
                    embedding_aggregator = embedding_aggregator.embedding_aggregator
                    embedding_aggregator.eval().requires_grad_(False)
                    if args.embedding_aggregator.args[i].grad[0] == "all":
                        embedding_aggregator.train().requires_grad_(True)
                    for layer in embedding_aggregator.named_modules():
                        if layer[0] in args.embedding_aggregator.args[i].grad:
                            layer[1].train().requires_grad_()
                    self.embedding_aggregator.append(embedding_aggregator)
                else:
                    self.embedding_aggregator.append(__E__(**args.embedding_aggregator.args[i]))
                    ckpt = args.embedding_aggregator.args[i].get("checkpoint", None)
                    if ckpt is not None:
                        try:
                            self.embedding_aggregator[i].load_state_dict(torch.load(ckpt)["state_dict"], strict=False)
                        except:
                            self.embedding_aggregator[i].load_state_dict(torch.load(ckpt), strict=False)

            self.embedding_aggregator = nn.ModuleList(self.embedding_aggregator)
        else:
            print("embedding_aggregator.__target__ should be a list")
            raise TypeError

        # loading Embedding Mixer
        __M_pkg__ = ".".join(args.embedding_mixer.__target__.split(".")[:-1])
        __M_name__ = args.embedding_mixer.__target__.split(".")[-1]
        __M__ = importlib.import_module(__M_pkg__).__getattribute__(__M_name__)
        self.embedding_mixer = __M__(**args.embedding_mixer)

        # loading Loss Function & Classifier
        __L_pkg__ = ".".join(args.loss_function.__target__.split(".")[:-1])
        __L_name__ = args.loss_function.__target__.split(".")[-1]
        __L__ = importlib.import_module(__L_pkg__).__getattribute__(__L_name__)
        self.loss_function = __L__(**args.loss_function)

        # loading Optimizer
        __O_pkg__ = ".".join(args.optimizer.__target__.split(".")[:-1])
        __O_name__ = args.optimizer.__target__.split(".")[-1]
        __O__ = importlib.import_module(__O_pkg__).__getattribute__(__O_name__)
        self.optim = __O__

        # loading Learning Rate Scheduler
        __S_pkg__ = ".".join(args.optimizer.lr_scheduler.__target__.split(".")[:-1])
        __S_name__ = args.optimizer.lr_scheduler.__target__.split(".")[-1]
        __S__ = importlib.import_module(__S_pkg__).__getattribute__(__S_name__)
        self.lr_sh = __S__

        try:
            if args.state_dict.split('.')[-1] == "ckpt":
                self.load_state_dict(torch.load(args.state_dict)["state_dict"])
            else:
                self.load_state_dict(torch.load(args.state_dict), strict=False)
        except Exception:
            print("XY1 load ckpt failed")
        self.count = 0

    def _video_downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4), device=self.args.device)
        out = torch.cat([out.data, zero_pads], dim=1)
        return out

    def _make_layer_face(self, block, out_channels, blocks, stride=(1, 1)):
        layers = [block(self.face_in_channels, out_channels, stride, downsample=True)]
        self.face_in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(self.face_in_channels, self.face_in_channels))
        return nn.Sequential(*layers)

    def _make_layer_video(self, block, blocks, planes, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.video_in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._video_downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.video_in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(inplanes=self.video_in_planes,
                  planes=planes,
                  stride=stride,
                  cardinality=self.video_cardinality,
                  downsample=downsample,
                  input_shape=self.video_STDA_input_shape))
        self.video_in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes=self.video_in_planes,
                                planes=planes,
                                stride=1,
                                cardinality=self.video_cardinality,
                                input_shape=[self.video_STDA_input_shape[0] * block.expansion,
                                             int(np.ceil(self.video_STDA_input_shape[1] / stride)),
                                             int(np.ceil(self.video_STDA_input_shape[2] / stride)),
                                             int(np.ceil(self.video_STDA_input_shape[3] / stride))]))

        return nn.Sequential(*layers)

    def forward(self, inpt, **kwargs) -> Any:
        # region    # <--- RawNet ---> #
        nb_samp = inpt[2].shape[0]
        len_seq = inpt[2].shape[1]
        x = self.voice_ln(inpt[2])
        x = x.view(nb_samp, 1, len_seq)
        x = F.max_pool1d(torch.abs(self.voice_first_conv(x)), 3)
        x = self.voice_first_bn(x)
        x = self.voice_lrelu_keras(x)
        x = self.voice_block0(x)
        x = self.voice_block1(x)
        x = self.voice_block2(x)
        x = self.voice_block3(x)
        x = self.voice_block4(x)
        x = self.voice_block5(x)
        x = self.voice_bn_before_gru(x)
        x = self.voice_lrelu_keras(x)
        x = x.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
        self.voice_gru.flatten_parameters()
        x, _ = self.voice_gru(x)
        x = x[:, -1, :]
        x = self.voice_fc1_gru(x)
        # endregion    # <--- RawNet ---> #

        # region    # <--- II-Net ---> #
        y = self.face_conv1(inpt[0])
        y = self.face_bn1(y)
        y = self.face_prelu(y)
        y = self.face_layer1(y)
        y = self.face_layer2(y)
        y = self.face_layer3(y)
        y = self.face_layer4(y)
        y = self.face_bn2(y)
        y = self.face_dropout(y)
        # endregion # <--- II-Net ---> #

        # region    # <--- Motion ---> #
        z = self.video_conv1(inpt[1])
        z = self.video_bn1(z)
        z = self.video_relu(z)
        z = self.video_maxpool(z)
        z = self.video_layer1(z)
        z = self.video_layer2(z)
        z = self.video_layer3(z)
        z = self.video_layer4(z)
        # endregion # <--- Motion ---> #

        embed_x = self.embedding_aggregator[2](x)
        embed_y = self.embedding_aggregator[0](y)
        embed_z = self.embedding_aggregator[1](z)
        embed = self.embedding_mixer([embed_y, embed_z, embed_x])
        return embed

    def configure_optimizers(self):
        param = filter(lambda p: p.requires_grad, self.parameters())
        opt = self.optim(param, **self.args.optimizer.args)
        if self.args.optimizer.lr_scheduler.args.last_epoch is None:
            if self.args.optimizer.lr_scheduler.interval == "step":
                self.args.optimizer.lr_scheduler.args.last_epoch = self.global_step - 1
            else:
                self.args.optimizer.lr_scheduler.args.last_epoch = self.current_epoch - 1
        lr_sh = self.lr_sh(opt, **self.args.optimizer.lr_scheduler.args)
        return [opt], [{'interval': self.args.optimizer.lr_scheduler.interval, 'scheduler': lr_sh}]

    def training_step(self, batch, batch_idx):
        x, y = batch
        embeddings = self(x)
        loss = self.loss_function(embeddings, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, label, pathes = batch
        embeddings = self(x)
        embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings, pathes

    def validation_epoch_end(self, val_step_outputs: List[Any]) -> None:
        # test REID
        embeds = torch.cat([x[0] for x in val_step_outputs])
        pathes = list(np.concatenate([x[1] for x in val_step_outputs], 0))
        d_embeddings = {}
        if not len(pathes) == len(embeds):
            print(len(pathes), len(embeds))
            exit()
        for k, v in zip(pathes, embeds):
            try:
                key = k[:-10]
            except:
                key = k[0][:-10]
            d_embeddings[key] = torch.squeeze(v).cpu()
        y_score = []  # score for each sample
        y = []  # label for each sample
        with open(self.args.trial_path, 'r') as f:
            l_trial = f.readlines()
        for line in l_trial:
            trg, utt_a, utt_b = line.strip().split(' ')
            y.append(int(trg))
            y_score.append(torch.cosine_similarity(d_embeddings[utt_a[:-10]], d_embeddings[utt_b[:-10]], dim=0))
        y_score = torch.tensor(y_score)
        tuned_threshold, eer, fpr, fnr = tune_threshold_from_score(y_score, y, [1, 0.1])
        fnrs, fprs, thresholds = compute_error_rates(y_score, y)
        min_dcf, min_c_det_threshold = compute_min_dcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        try:
            self.log_dict({"EER": eer, "MinDCF": min_dcf}, on_step=False, on_epoch=True)
        except:
            print("Epoch {}: EER logging failed!".format(self.current_epoch))
        print("epoch {} step {}: EER reaches {}%".format(self.current_epoch, self.global_step, eer))

    def test_step(self, batch, batch_idx):
        self.count += 1
        if self.count == 113:
            print("113")
        x, label, pathes = batch
        union_embeddings = self(x)
        union_embeddings = torch.nn.functional.normalize(union_embeddings)
        return union_embeddings, pathes

    def test_epoch_end(self, outputs: List[Any]) -> None:
        # test REID
        result = ["epoch:{}".format(self.current_epoch),
                  "step:{}".format(self.global_step),
                  "ckpt:{}".format(self.args.checkpoint)]

        union_embeds = torch.cat([x[0] for x in outputs])
        pathes = list(np.concatenate([x[-1] for x in outputs], 0))
        d_embeddings = {}
        if not len(pathes) == len(union_embeds):
            print(len(pathes), len(union_embeds))
            exit()
        for k, v1 in zip(pathes, union_embeds):
            try:
                key = k[:-10]
            except:
                key = k[0][:-10]
            d_embeddings[key] = torch.squeeze(v1).cpu()

        y_score = []  # score for each sample
        y = []  # label for each sample
        with open(self.args.trial_path, 'r') as f:
            l_trial = f.readlines()
        for line in tqdm(l_trial, desc="cal_score"):
            trg, utt_a, utt_b = line.strip().split(' ')
            y.append(int(trg))
            y_score.append(torch.cosine_similarity(d_embeddings[utt_a[:-10]], d_embeddings[utt_b[:-10]], dim=0))
        y_score = torch.tensor(y_score)
        tuned_threshold, eer, fpr, fnr = tune_threshold_from_score(y_score, y, [1, 0.1])
        fnrs, fprs, thresholds = compute_error_rates(y_score, y)
        min_dcf, min_c_det_threshold = compute_min_dcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        self.log_dict({"EER": eer, "MinDCF": min_dcf}, on_step=False, on_epoch=True)
        print("Calculated by compute_min_cost:\nEER = {0}%\nminDCF = {1}\n".format(eer, min_dcf))
        print()
        result.append(
            "Epoch {} step {}:\nEER = {}%\nminDCF = {}".format(self.current_epoch, self.global_step, eer, min_dcf))
        with open("result.txt", 'w') as f:
            f.write("\n".join(result))


class MFFN_Face_Audio(MFFN_BASE):
    def __init__(self, args: DictConfig):
        super(MFFN_Face_Audio, self).__init__(args)
        # self.attx_fc = nn.Linear(1024, 512)
        # self.atty_fc = nn.Identity()
        # self.attz_fc = nn.Linear(2048, 512)
        print("Using MFFN_Face_Audio")

    def forward(self, inpt, **kwargs) -> Any:
        # region    # <--- RawNet ---> #
        nb_samp = inpt[2].shape[0]
        len_seq = inpt[2].shape[1]
        x = self.voice_ln(inpt[2])
        x = x.view(nb_samp, 1, len_seq)
        x = F.max_pool1d(torch.abs(self.voice_first_conv(x)), 3)
        x = self.voice_first_bn(x)
        x = self.voice_lrelu_keras(x)
        x = self.voice_block0(x)
        x = self.voice_block1(x)
        x = self.voice_block2(x)
        x = self.voice_block3(x)
        x = self.voice_block4(x)
        x = self.voice_block5(x)
        x = self.voice_bn_before_gru(x)
        x = self.voice_lrelu_keras(x)
        x = x.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
        self.voice_gru.flatten_parameters()
        x, _ = self.voice_gru(x)
        x = x[:, -1, :]
        x = self.voice_fc1_gru(x)
        # endregion # <--- RA ---> #

        # region    # <--- II-Net ---> #
        y = self.face_conv1(inpt[0])
        y = self.face_bn1(y)
        y = self.face_prelu(y)
        y = self.face_layer1(y)
        y = self.face_layer2(y)
        y = self.face_layer3(y)
        y = self.face_layer4(y)
        y = self.face_bn2(y)
        y = self.face_dropout(y)
        # endregion # <--- II-Net ---> #

        # # region    # <--- Motion ---> #
        # z = self.video_conv1(inpt[1])
        # z = self.video_bn1(z)
        # z = self.video_relu(z)
        # z = self.video_maxpool(z)
        #
        # z = self.video_layer1(z)
        # z = self.video_layer2(z)
        # z = self.video_layer3(z)
        # z = self.video_layer4(z)
        # # endregion # <--- Motion ---> #

        embed_x = self.embedding_aggregator[1](x)
        embed_y = self.embedding_aggregator[0](y)
        # embed_z = self.embedding_aggregator[1](z)

        # embed_x = self.attx_fc(embed_x)
        # embed_y = self.atty_fc(embed_y)
        # embed_z = self.attz_fc(embed_z)

        embed = self.embedding_mixer([embed_x, embed_y])
        return embed


class MFFN_Video_Audio(MFFN_BASE):
    def __init__(self, args: DictConfig):
        super(MFFN_Video_Audio, self).__init__(args)
        # self.attx_fc = nn.Linear(1024, 512)
        # self.atty_fc = nn.Identity()
        # self.attz_fc = nn.Linear(2048, 512)
        print("Using MFFN_Video_Audio")

    def forward(self, inpt, **kwargs) -> Any:
        # region    # <--- RawNet ---> #
        nb_samp = inpt[2].shape[0]
        len_seq = inpt[2].shape[1]
        x = self.voice_ln(inpt[2])
        x = x.view(nb_samp, 1, len_seq)
        x = F.max_pool1d(torch.abs(self.voice_first_conv(x)), 3)
        x = self.voice_first_bn(x)
        x = self.voice_lrelu_keras(x)
        x = self.voice_block0(x)
        x = self.voice_block1(x)
        x = self.voice_block2(x)
        x = self.voice_block3(x)
        x = self.voice_block4(x)
        x = self.voice_block5(x)
        x = self.voice_bn_before_gru(x)
        x = self.voice_lrelu_keras(x)
        x = x.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
        self.voice_gru.flatten_parameters()
        x, _ = self.voice_gru(x)
        x = x[:, -1, :]
        x = self.voice_fc1_gru(x)
        # endregion # <--- RA ---> #

        # # region    # <--- II-Net ---> #
        # y = self.face_conv1(inpt[0])
        # y = self.face_bn1(y)
        # y = self.face_prelu(y)
        # y = self.face_layer1(y)
        # y = self.face_layer2(y)
        # y = self.face_layer3(y)
        # y = self.face_layer4(y)
        # y = self.face_bn2(y)
        # y = self.face_dropout(y)
        # # endregion # <--- II-Net ---> #

        # region    # <--- Motion ---> #
        z = self.video_conv1(inpt[1])
        z = self.video_bn1(z)
        z = self.video_relu(z)
        z = self.video_maxpool(z)

        z = self.video_layer1(z)
        z = self.video_layer2(z)
        z = self.video_layer3(z)
        z = self.video_layer4(z)
        # endregion # <--- Motion ---> #

        embed_x = self.embedding_aggregator[1](x)
        # embed_y = self.embedding_aggregator[0](y)
        embed_z = self.embedding_aggregator[0](z)

        # embed_x = self.attx_fc(embed_x)
        # embed_y = self.atty_fc(embed_y)
        # embed_z = self.attz_fc(embed_z)

        embed = self.embedding_mixer([embed_x, embed_z])
        return embed


class MFFN_CBP1(MFFN_BASE):
    def __init__(self, args: DictConfig):
        super(MFFN_CBP1, self).__init__(args)
        self.attx_fc = nn.Linear(1024, 512)
        self.atty_fc = nn.Identity()
        self.attz_fc = nn.Linear(2048, 512)

        self.attxy_cbp = CompactBilinearPooling(512, 512, 1024)
        self.attxz_cbp = CompactBilinearPooling(512, 512, 1024)
        self.attyz_cbp = CompactBilinearPooling(512, 512, 1024)

    def forward(self, inpt, **kwargs) -> Any:
        # region    # <--- RawNet ---> #
        nb_samp = inpt[2].shape[0]
        len_seq = inpt[2].shape[1]
        x = self.voice_ln(inpt[2])
        x = x.view(nb_samp, 1, len_seq)
        x = F.max_pool1d(torch.abs(self.voice_first_conv(x)), 3)
        x = self.voice_first_bn(x)
        x = self.voice_lrelu_keras(x)
        x = self.voice_block0(x)
        x = self.voice_block1(x)
        x = self.voice_block2(x)
        x = self.voice_block3(x)
        x = self.voice_block4(x)
        x = self.voice_block5(x)
        x = self.voice_bn_before_gru(x)
        x = self.voice_lrelu_keras(x)
        x = x.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
        self.voice_gru.flatten_parameters()
        x, _ = self.voice_gru(x)
        x = x[:, -1, :]
        x = self.voice_fc1_gru(x)
        # endregion # <--- RA ---> #

        # region    # <--- II-Net ---> #
        y = self.face_conv1(inpt[0])
        y = self.face_bn1(y)
        y = self.face_prelu(y)
        y = self.face_layer1(y)
        y = self.face_layer2(y)
        y = self.face_layer3(y)
        y = self.face_layer4(y)
        y = self.face_bn2(y)
        y = self.face_dropout(y)
        # endregion # <--- II-Net ---> #

        # region    # <--- Motion ---> #
        z = self.video_conv1(inpt[1])
        z = self.video_bn1(z)
        z = self.video_relu(z)
        z = self.video_maxpool(z)

        z = self.video_layer1(z)
        z = self.video_layer2(z)
        z = self.video_layer3(z)
        z = self.video_layer4(z)
        # endregion # <--- Motion ---> #

        embed_x = self.embedding_aggregator[2](x)
        embed_y = self.embedding_aggregator[0](y)
        embed_z = self.embedding_aggregator[1](z)

        embed_x = self.attx_fc(embed_x)
        embed_y = self.atty_fc(embed_y)
        embed_z = self.attz_fc(embed_z)

        embed_xy = self.attxy_cbp(embed_x, embed_y)
        embed_xz = self.attxz_cbp(embed_x, embed_z)
        embed_yz = self.attyz_cbp(embed_y, embed_z)

        embed = self.embedding_mixer([embed_xy, embed_xz, embed_yz])
        return embed


class MFFN_MLB1(MFFN_BASE):
    def __init__(self, args: DictConfig):
        super(MFFN_MLB1, self).__init__(args)
        self.attx_fc = nn.Linear(1024, 512)
        self.atty_fc = nn.Identity()
        self.attz_fc = nn.Linear(2048, 512)

        self.attxy_mlb = nn.Linear(512, 512)
        self.attxz_mlb = nn.Linear(512, 512)
        self.attyz_mlb = nn.Linear(512, 512)

    def forward(self, inpt, **kwargs) -> Any:
        # region    # <--- RawNet ---> #
        nb_samp = inpt[2].shape[0]
        len_seq = inpt[2].shape[1]
        x = self.voice_ln(inpt[2])
        x = x.view(nb_samp, 1, len_seq)
        x = F.max_pool1d(torch.abs(self.voice_first_conv(x)), 3)
        x = self.voice_first_bn(x)
        x = self.voice_lrelu_keras(x)
        x = self.voice_block0(x)
        x = self.voice_block1(x)
        x = self.voice_block2(x)
        x = self.voice_block3(x)
        x = self.voice_block4(x)
        x = self.voice_block5(x)
        x = self.voice_bn_before_gru(x)
        x = self.voice_lrelu_keras(x)
        x = x.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
        self.voice_gru.flatten_parameters()
        x, _ = self.voice_gru(x)
        x = x[:, -1, :]
        x = self.voice_fc1_gru(x)
        # endregion # <--- RA ---> #

        # region    # <--- II-Net ---> #
        y = self.face_conv1(inpt[0])
        y = self.face_bn1(y)
        y = self.face_prelu(y)
        y = self.face_layer1(y)
        y = self.face_layer2(y)
        y = self.face_layer3(y)
        y = self.face_layer4(y)
        y = self.face_bn2(y)
        y = self.face_dropout(y)
        # endregion # <--- II-Net ---> #

        # region    # <--- Motion ---> #
        z = self.video_conv1(inpt[1])
        z = self.video_bn1(z)
        z = self.video_relu(z)
        z = self.video_maxpool(z)

        z = self.video_layer1(z)
        z = self.video_layer2(z)
        z = self.video_layer3(z)
        z = self.video_layer4(z)
        # endregion # <--- Motion ---> #

        embed_x = self.embedding_aggregator[2](x)
        embed_y = self.embedding_aggregator[0](y)
        embed_z = self.embedding_aggregator[1](z)

        embed_x = self.attx_fc(embed_x)
        embed_y = self.atty_fc(embed_y)
        embed_z = self.attz_fc(embed_z)

        embed_xy = embed_x * embed_y
        embed_xz = embed_x * embed_z
        embed_yz = embed_y * embed_z

        embed_xy = F.gelu(self.attxy_mlb(embed_xy))
        embed_xz = F.gelu(self.attxz_mlb(embed_xz))
        embed_yz = F.gelu(self.attyz_mlb(embed_yz))

        embed = self.embedding_mixer([embed_xy, embed_xz, embed_yz])
        return embed


class MFFN_attVALD1(MFFN_BASE):
    def __init__(self, args: DictConfig):
        super(MFFN_attVALD1, self).__init__(args)
        self.example_input_array = [[torch.rand((2, 16, 3, args.image_size, args.image_size)).to(args.device),
                                     torch.rand((2, 3, 16, args.image_size, args.image_size)).to(args.device),
                                     torch.rand((2, args.nb_samp)).to(args.device)]]
        self.attx_fc = nn.Linear(1024, 512)
        self.atty_fc = nn.Identity()
        self.attz_fc = nn.Linear(2048, 512)

        self.att_vlad = AttVlad(8, 512)
        self.att_mlma = MLMA(512, None)

    def forward(self, inpt, **kwargs) -> Any:
        embeds = []

        # region    # <--- RawNet ---> #
        nb_samp = inpt[2].shape[0]
        len_seq = inpt[2].shape[1]
        x = self.voice_ln(inpt[2])
        x = x.view(nb_samp, 1, len_seq)
        x = F.max_pool1d(torch.abs(self.voice_first_conv(x)), 3)
        x = self.voice_first_bn(x)
        x = self.voice_lrelu_keras(x)
        x = self.voice_block0(x)
        x = self.voice_block1(x)
        x = self.voice_block2(x)
        x = self.voice_block3(x)
        x = self.voice_block4(x)
        x = self.voice_block5(x)
        x = self.voice_bn_before_gru(x)
        x = self.voice_lrelu_keras(x)
        x = x.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
        self.voice_gru.flatten_parameters()
        x, _ = self.voice_gru(x)
        x = x[:, -1, :]
        x = self.voice_fc1_gru(x)
        embed_x = self.embedding_aggregator[2](x)
        embeds.append(self.attx_fc(embed_x))  # Bx512
        # endregion # <--- RA ---> #

        # region    # <--- II-Net ---> #
        y_embeds = []
        for i in range(inpt[0].shape[0]):
            y = self.face_conv1(inpt[0][i])
            y = self.face_bn1(y)
            y = self.face_prelu(y)
            y = self.face_layer1(y)
            y = self.face_layer2(y)
            y = self.face_layer3(y)
            y = self.face_layer4(y)
            y = self.face_bn2(y)
            y = self.face_dropout(y)
            embed_y = self.embedding_aggregator[0](y)
            y_embeds.append(self.atty_fc(embed_y))  # BxKx512
        embeds.extend(list(torch.split(self.att_vlad(torch.stack(y_embeds).transpose(1, 2)), 1, 1)))
        # endregion # <--- II-Net ---> #

        # region    # <--- Motion ---> #
        z = self.video_conv1(inpt[1])
        z = self.video_bn1(z)
        z = self.video_relu(z)
        z = self.video_maxpool(z)

        z = self.video_layer1(z)
        z = self.video_layer2(z)
        z = self.video_layer3(z)
        z = self.video_layer4(z)
        embed_z = self.embedding_aggregator[1](z)
        embeds.append(self.attz_fc(embed_z))  # Bx512
        # endregion # <--- Motion ---> #

        embeds = list(map(lambda x: x.squeeze(1), embeds))
        embed = self.embedding_mixer(embeds)
        return embed


class MFFN_Transformer(MFFN_BASE):
    def __init__(self, args: DictConfig):
        super(MFFN_Transformer, self).__init__(args)

        att_channel = args.cross_attention.middle_channels

        self.attxy_voice = nn.Conv1d(80, att_channel, 1)
        self.attxy_face = nn.Conv1d(14 * 14, att_channel, 1)
        self.attxy_ff = nn.Sequential(nn.Conv1d(256, 512, 1),
                                      nn.ReLU(),
                                      nn.Conv1d(512, 256, 1))
        self.attyx_voice = nn.Conv1d(80, att_channel, 1)
        self.attyx_face = nn.Conv1d(14 * 14, att_channel, 1)
        self.attyx_ff = nn.Sequential(nn.Conv2d(256, 512, (1, 1)),
                                      nn.ReLU(),
                                      nn.Conv2d(512, 256, (1, 1)))

        self.attxy_voice2 = nn.Conv1d(80, att_channel, 1)
        self.attxy_face2 = nn.Conv1d(14 * 14, att_channel, 1)
        self.attxy_ff2 = nn.Sequential(nn.Conv1d(256, 512, 1),
                                       nn.ReLU(),
                                       nn.Conv1d(512, 256, 1))
        self.attyx_voice2 = nn.Conv1d(80, att_channel, 1)
        self.attyx_face2 = nn.Conv1d(14 * 14, att_channel, 1)
        self.attyx_ff2 = nn.Sequential(nn.Conv2d(256, 512, (1, 1)),
                                       nn.ReLU(),
                                       nn.Conv2d(512, 256, (1, 1)))

        try:
            if args.state_dict.split('.')[-1] == "ckpt":
                self.load_state_dict(torch.load(args.state_dict)["state_dict"])
            else:
                self.load_state_dict(torch.load(args.state_dict), strict=False)
            print("Transformer load ckpt success")
        except Exception:
            print("Transformer load ckpt failed")

    def forward(self, inpt, **kwargs) -> Any:
        # region    # <--- before ---> #
        nb_samp = inpt[2].shape[0]
        len_seq = inpt[2].shape[1]
        x = self.voice_ln(inpt[2])
        x = x.view(nb_samp, 1, len_seq)
        x = F.max_pool1d(torch.abs(self.voice_first_conv(x)), 3)
        x = self.voice_first_bn(x)
        x = self.voice_lrelu_keras(x)
        x = self.voice_block0(x)
        x = self.voice_block1(x)
        x = self.voice_block2(x)
        x = self.voice_block3(x)
        x = self.voice_block4(x)

        y = self.face_conv1(inpt[0])
        y = self.face_bn1(y)
        y = self.face_prelu(y)
        y = self.face_layer1(y)
        y = self.face_layer2(y)
        y = self.face_layer3(y)

        z = self.video_conv1(inpt[1])
        z = self.video_bn1(z)
        z = self.video_relu(z)
        z = self.video_maxpool(z)
        z = self.video_layer1(z)
        z = self.video_layer2(z)
        # endregion # <--- before ---> #

        attxy_voice = self.attxy_voice(x.transpose(1, 2))  # 256*80
        attxy_face = self.attxy_face(y.view(y.shape[0], y.shape[1], -1).transpose(1, 2))  # 256*14*14
        attxy = torch.softmax(torch.matmul(attxy_face.transpose(1, 2), attxy_voice).transpose(1, 2), -1) / np.sqrt(
            256)  # 256*256
        attxy2 = torch.layer_norm(x + torch.matmul(attxy, x), (256, 80))  # 256*80
        attxy_ff = torch.layer_norm(self.attxy_ff(attxy2) + attxy2, (256, 80))  # 256*80
        attyx_voice = self.attyx_voice(x.transpose(1, 2))  # 256*80
        attyx_face = self.attyx_face(y.view(y.shape[0], y.shape[1], -1).transpose(1, 2))  # 256*14*14
        attyx = torch.softmax(torch.matmul(attyx_voice.transpose(1, 2), attyx_face).transpose(1, 2), -1) / np.sqrt(
            256)  # 256*256
        attyx2 = torch.layer_norm(y + torch.matmul(attyx, y.view(y.shape[0], y.shape[1], -1)).view(-1, 256, 14, 14),
                                  (256, 14, 14))  # 256*14*14
        attyx_ff = torch.layer_norm(self.attyx_ff(attyx2) + attyx2, (256, 14, 14))  # 256*14*14

        attxy_voice = self.attxy_voice2(attxy_ff.transpose(1, 2))  # 256*80
        attxy_face = self.attxy_face2(
            attyx_ff.view(attyx_ff.shape[0], attyx_ff.shape[1], -1).transpose(1, 2))  # 256*14*14
        attxy = torch.softmax(torch.matmul(attxy_face.transpose(1, 2), attxy_voice).transpose(1, 2), -1) / np.sqrt(
            256)  # 256*256
        attxy2 = torch.layer_norm(attxy_ff + torch.matmul(attxy, attxy_ff), (256, 80))  # 256*80
        attxy_ff2 = torch.layer_norm(self.attxy_ff2(attxy2) + attxy2, (256, 80))  # 256*80
        attyx_voice = self.attyx_voice2(attxy_ff.transpose(1, 2))  # 256*80
        attyx_face = self.attyx_face2(
            attyx_ff.view(attyx_ff.shape[0], attyx_ff.shape[1], -1).transpose(1, 2))  # 256*14*14
        attyx = torch.softmax(torch.matmul(attyx_voice.transpose(1, 2), attyx_face).transpose(1, 2), -1) / np.sqrt(
            256)  # 256*256
        attyx2 = torch.layer_norm(
            attyx_ff + torch.matmul(attyx, attyx_ff.view(attyx_ff.shape[0], attyx_ff.shape[1], -1)).view(-1, 256, 14,
                                                                                                         14),
            (256, 14, 14))  # 256*14*14
        attyx_ff2 = torch.layer_norm(self.attyx_ff2(attyx2) + attyx2, (256, 14, 14))  # 256*14*14

        # x = attxy_ff
        # y = attyx_ff

        # region    # <--- after ---> #

        x = self.voice_block5(attxy_ff2)
        x = self.voice_bn_before_gru(x)
        x = self.voice_lrelu_keras(x)
        x = x.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
        self.voice_gru.flatten_parameters()
        x, _ = self.voice_gru(x)
        x = x[:, -1, :]
        x = self.voice_fc1_gru(x)

        y = self.face_layer4(attyx_ff2)
        y = self.face_bn2(y)
        y = self.face_dropout(y)

        z = self.video_layer3(z)
        z = self.video_layer4(z)

        # endregion # <--- after ---> #

        embed_x = self.embedding_aggregator[2](x)
        embed_y = self.embedding_aggregator[0](y)
        embed_z = self.embedding_aggregator[1](z)
        embed = self.embedding_mixer([embed_y, embed_z, embed_x])
        return embed


class MFFN_ICODE1(MFFN_BASE):
    def __init__(self, args: DictConfig):
        super(MFFN_ICODE1, self).__init__(args)
        self.attx_fc = nn.Linear(1024, 1022)
        self.atty_fc = nn.Linear(512, 1022)
        self.attz_fc = nn.Linear(2048, 1022)

        encoder_layer = nn.TransformerEncoderLayer(d_model=3072, nhead=8)
        self.att_merge = torch.nn.TransformerEncoder(encoder_layer, num_layers=8)

        self.attxx_icode = nn.MultiheadAttention(1024, 8)
        self.attyy_icode = nn.MultiheadAttention(1024, 8)
        self.attzz_icode = nn.MultiheadAttention(1024, 8)

        self.attxy_icode = nn.MultiheadAttention(1024, 8)
        self.attyz_icode = nn.MultiheadAttention(1024, 8)
        self.attzx_icode = nn.MultiheadAttention(1024, 8)

        self.attxy_fc = nn.Linear(1024, 1024)
        self.attyz_fc = nn.Linear(1024, 1024)
        self.attzx_fc = nn.Linear(1024, 1024)

    def forward(self, inpt, **kwargs) -> Any:
        # region    # <--- RawNet ---> #
        nb_samp = inpt[2].shape[0]
        len_seq = inpt[2].shape[1]
        x = self.voice_ln(inpt[2])
        x = x.view(nb_samp, 1, len_seq)
        x = F.max_pool1d(torch.abs(self.voice_first_conv(x)), 3)
        x = self.voice_first_bn(x)
        x = self.voice_lrelu_keras(x)
        x = self.voice_block0(x)
        x = self.voice_block1(x)
        x = self.voice_block2(x)
        x = self.voice_block3(x)
        x = self.voice_block4(x)
        x = self.voice_block5(x)
        x = self.voice_bn_before_gru(x)
        x = self.voice_lrelu_keras(x)
        x = x.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
        self.voice_gru.flatten_parameters()
        x, _ = self.voice_gru(x)
        x = x[:, -1, :]
        x = self.voice_fc1_gru(x)
        # endregion # <--- RA ---> #

        # region    # <--- II-Net ---> #
        y = self.face_conv1(inpt[0])
        y = self.face_bn1(y)
        y = self.face_prelu(y)
        y = self.face_layer1(y)
        y = self.face_layer2(y)
        y = self.face_layer3(y)
        y = self.face_layer4(y)
        y = self.face_bn2(y)
        y = self.face_dropout(y)
        # endregion # <--- II-Net ---> #

        # region    # <--- Motion ---> #
        z = self.video_conv1(inpt[1])
        z = self.video_bn1(z)
        z = self.video_relu(z)
        z = self.video_maxpool(z)

        z = self.video_layer1(z)
        z = self.video_layer2(z)
        z = self.video_layer3(z)
        z = self.video_layer4(z)
        # endregion # <--- Motion ---> #

        embed_x = self.embedding_aggregator[2](x)
        embed_y = self.embedding_aggregator[0](y)
        embed_z = self.embedding_aggregator[1](z)

        embed_x = self.attx_fc(embed_x)
        embed_y = self.atty_fc(embed_y)
        embed_z = self.attz_fc(embed_z)

        embed_x = torch.cat(
            [embed_x, torch.repeat_interleave(torch.tensor([[0, 0]]), embed_x.shape[0], dim=0).to("cuda")],
            dim=1).unsqueeze(1)
        embed_y = torch.cat(
            [embed_y, torch.repeat_interleave(torch.tensor([[0, 1]]), embed_x.shape[0], dim=0).to("cuda")],
            dim=1).unsqueeze(1)
        embed_z = torch.cat(
            [embed_z, torch.repeat_interleave(torch.tensor([[1, 1]]), embed_x.shape[0], dim=0).to("cuda")],
            dim=1).unsqueeze(1)

        embed_cat = torch.cat([embed_x, embed_y, embed_z], dim=2)
        embed_merge = self.att_merge(embed_cat)

        embed_x = embed_merge[:, :, :1024]
        embed_y = embed_merge[:, :, 1024:2048]
        embed_z = embed_merge[:, :, 2048:]

        embed_xx = self.attxx_icode(embed_x, embed_x, embed_x)[0]
        embed_yy = self.attyy_icode(embed_y, embed_y, embed_y)[0]
        embed_zz = self.attyy_icode(embed_z, embed_z, embed_z)[0]

        embed_xy = self.attxy_icode(embed_yy, embed_xx, embed_xx)[0]
        embed_yz = self.attyz_icode(embed_zz, embed_yy, embed_yy)[0]
        embed_zx = self.attzx_icode(embed_xx, embed_zz, embed_zz)[0]

        xy = torch.layer_norm(embed_xy.squeeze() + self.attxy_fc(embed_xy.squeeze()), (1024,))
        yz = torch.layer_norm(embed_yz.squeeze() + self.attyz_fc(embed_yz.squeeze()), (1024,))
        zx = torch.layer_norm(embed_zx.squeeze() + self.attzx_fc(embed_zx.squeeze()), (1024,))

        embed = self.embedding_mixer([xy, zx, yz])
        return embed


class MFFN_ICODE2(MFFN_BASE):
    def __init__(self, args: DictConfig):
        super(MFFN_ICODE2, self).__init__(args)
        self.attx_fc = nn.Linear(1024, 510)
        self.atty_fc = nn.Linear(512, 510)
        self.attz_fc = nn.Linear(2048, 510)

        encoder_layer = nn.TransformerEncoderLayer(d_model=1536, nhead=8)
        self.att_merge = torch.nn.TransformerEncoder(encoder_layer, num_layers=8)

        self.attxx_icode = nn.MultiheadAttention(512, 8)
        self.attyy_icode = nn.MultiheadAttention(512, 8)
        self.attzz_icode = nn.MultiheadAttention(512, 8)

        self.attxy_icode = nn.MultiheadAttention(512, 8)
        self.attyz_icode = nn.MultiheadAttention(512, 8)
        self.attzx_icode = nn.MultiheadAttention(512, 8)

        self.attxy_fc = nn.Linear(512, 512)
        self.attyz_fc = nn.Linear(512, 512)
        self.attzx_fc = nn.Linear(512, 512)

    def forward(self, inpt, **kwargs) -> Any:
        # region    # <--- RawNet ---> #
        nb_samp = inpt[2].shape[0]
        len_seq = inpt[2].shape[1]
        x = self.voice_ln(inpt[2])
        x = x.view(nb_samp, 1, len_seq)
        x = F.max_pool1d(torch.abs(self.voice_first_conv(x)), 3)
        x = self.voice_first_bn(x)
        x = self.voice_lrelu_keras(x)
        x = self.voice_block0(x)
        x = self.voice_block1(x)
        x = self.voice_block2(x)
        x = self.voice_block3(x)
        x = self.voice_block4(x)
        x = self.voice_block5(x)
        x = self.voice_bn_before_gru(x)
        x = self.voice_lrelu_keras(x)
        x = x.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
        self.voice_gru.flatten_parameters()
        x, _ = self.voice_gru(x)
        x = x[:, -1, :]
        x = self.voice_fc1_gru(x)
        # endregion # <--- RA ---> #

        # region    # <--- II-Net ---> #
        y = self.face_conv1(inpt[0])
        y = self.face_bn1(y)
        y = self.face_prelu(y)
        y = self.face_layer1(y)
        y = self.face_layer2(y)
        y = self.face_layer3(y)
        y = self.face_layer4(y)
        y = self.face_bn2(y)
        y = self.face_dropout(y)
        # endregion # <--- II-Net ---> #

        # region    # <--- Motion ---> #
        z = self.video_conv1(inpt[1])
        z = self.video_bn1(z)
        z = self.video_relu(z)
        z = self.video_maxpool(z)

        z = self.video_layer1(z)
        z = self.video_layer2(z)
        z = self.video_layer3(z)
        z = self.video_layer4(z)
        # endregion # <--- Motion ---> #

        embed_x = self.embedding_aggregator[2](x)
        embed_y = self.embedding_aggregator[0](y)
        embed_z = self.embedding_aggregator[1](z)

        embed_x = self.attx_fc(embed_x)
        embed_y = self.atty_fc(embed_y)
        embed_z = self.attz_fc(embed_z)

        embed_x = torch.cat(
            [embed_x, torch.repeat_interleave(torch.tensor([[0, 0]]), embed_x.shape[0], dim=0).to("cuda")],
            dim=1).unsqueeze(1)
        embed_y = torch.cat(
            [embed_y, torch.repeat_interleave(torch.tensor([[0, 1]]), embed_x.shape[0], dim=0).to("cuda")],
            dim=1).unsqueeze(1)
        embed_z = torch.cat(
            [embed_z, torch.repeat_interleave(torch.tensor([[1, 1]]), embed_x.shape[0], dim=0).to("cuda")],
            dim=1).unsqueeze(1)

        embed_cat = torch.cat([embed_x, embed_y, embed_z], dim=2)
        embed_merge = self.att_merge(embed_cat)

        embed_x = embed_merge[:, :, :512]
        embed_y = embed_merge[:, :, 512:1024]
        embed_z = embed_merge[:, :, 1024:]

        embed_xx = self.attxx_icode(embed_x, embed_x, embed_x)[0]
        embed_yy = self.attyy_icode(embed_y, embed_y, embed_y)[0]
        embed_zz = self.attyy_icode(embed_z, embed_z, embed_z)[0]

        embed_xy = self.attxy_icode(embed_yy, embed_xx, embed_xx)[0]
        embed_yz = self.attyz_icode(embed_zz, embed_yy, embed_yy)[0]
        embed_zx = self.attzx_icode(embed_xx, embed_zz, embed_zz)[0]

        xy = torch.layer_norm(embed_xy.squeeze() + self.attxy_fc(embed_xy.squeeze()), (512,))
        yz = torch.layer_norm(embed_yz.squeeze() + self.attyz_fc(embed_yz.squeeze()), (512,))
        zx = torch.layer_norm(embed_zx.squeeze() + self.attzx_fc(embed_zx.squeeze()), (512,))

        embed = self.embedding_mixer([xy, zx, yz])
        return embed


# region    # <--- ML_MDA ---> #

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Flatten(nn.Module):
    def forward(self, x):
        return (x.view(x.size(0), x.size(1)))


class face_module(nn.Module):
    def __init__(self, fusion_layer=4, pool_dim=512):
        super(face_module, self).__init__()

        self.face = resnet18(pretrained=True)

        self.fusion_layer = fusion_layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)

        layer0 = [self.face.conv1, self.face.bn1, self.face.relu, self.face.maxpool]
        self.layer_dict = {"layer0": nn.Sequential(*layer0), "layer1": self.face.layer1,
                           "layer2": self.face.layer2, "layer3": self.face.layer3,
                           "layer4": self.face.layer4,
                           "layer5": self.avgpool, "layer6": Flatten(), "layer7": self.bottleneck}

    def forward(self, x, with_features=False):
        for i in range(0, self.fusion_layer):
            if i == 5:
                backbone_feat = x
                x_pool = self.layer_dict["layer5"](x)
                x_pool = self.layer_dict["layer6"](x_pool)
                feat = self.layer_dict["layer7"](x_pool)
                if with_features:
                    return [x_pool, feat, backbone_feat]
                return [x_pool, feat]
            if i < 5:
                x = self.layer_dict["layer" + str(i)](x)

        return x


class body_module(nn.Module):
    def __init__(self, fusion_layer=4, pool_dim=512):
        super(body_module, self).__init__()

        self.body = resnet18(pretrained=True)

        self.fusion_layer = fusion_layer

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)

        layer0 = [self.body.conv1, self.body.bn1, self.body.relu, self.body.maxpool]
        self.layer_dict = {"layer0": nn.Sequential(*layer0), "layer1": self.body.layer1,
                           "layer2": self.body.layer2, "layer3": self.body.layer3,
                           "layer4": self.body.layer4,
                           "layer5": self.avgpool, "layer6": Flatten(), "layer7": self.bottleneck}

    def forward(self, x, with_features=False):
        for i in range(0, self.fusion_layer):
            if i == 5:
                backbone_feat = x
                x_pool = self.layer_dict["layer5"](x)
                x_pool = self.layer_dict["layer6"](x_pool)
                feat = self.layer_dict["layer7"](x_pool)
                if with_features:
                    return [x_pool, feat, backbone_feat]
                return [x_pool, feat]
            if i < 5:
                x = self.layer_dict["layer" + str(i)](x)
        return x


class voice_module(nn.Module):
    def __init__(self, fusion_layer=4, pool_dim=512):
        super(voice_module, self).__init__()

        self.voice = resnet18(pretrained=True)

        self.fusion_layer = fusion_layer

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)

        layer0 = [self.voice.conv1, self.voice.bn1, self.voice.relu, self.voice.maxpool]
        self.layer_dict = {"layer0": nn.Sequential(*layer0), "layer1": self.voice.layer1,
                           "layer2": self.voice.layer2, "layer3": self.voice.layer3,
                           "layer4": self.voice.layer4,
                           "layer5": self.avgpool, "layer6": Flatten(), "layer7": self.bottleneck}

    def forward(self, x, with_features=False):
        for i in range(0, self.fusion_layer):
            if i == 5:
                backbone_feat = x
                x_pool = self.layer_dict["layer5"](x)
                x_pool = self.layer_dict["layer6"](x_pool)
                feat = self.layer_dict["layer7"](x_pool)
                if with_features:
                    return [x_pool, feat, backbone_feat]
                return [x_pool, feat]
            if i < 5:
                x = self.layer_dict["layer" + str(i)](x)
        return x


class shared_resnet(nn.Module):
    def __init__(self, fusion_layer=4, pool_dim=512):
        super(shared_resnet, self).__init__()

        self.fusion_layer = fusion_layer

        model_base = resnet18(pretrained=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.base = model_base

        layer0 = [self.base.conv1, self.base.bn1, self.base.relu, self.base.maxpool]
        self.layer_dict = {"layer0": nn.Sequential(*layer0), "layer1": self.base.layer1,
                           "layer2": self.base.layer2, "layer3": self.base.layer3, "layer4": self.base.layer4,
                           "layer5": self.avgpool, "layer6": Flatten(), "layer7": self.bottleneck}

    def forward(self, x):

        for i in range(self.fusion_layer, 6):
            if i < 5:
                x = self.layer_dict["layer" + str(i)](x)
            else:
                x_pool = self.layer_dict["layer5"](x)
                x_pool = self.layer_dict["layer6"](x_pool)
                feat = self.layer_dict["layer7"](x_pool)
                return [x_pool, feat]


class ML_MDA(pl.LightningModule):
    def __init__(self, args: DictConfig):
        super().__init__()
        self.example_input_array = [[torch.rand((2, 3, args.image_size, args.image_size)).to(args.device),
                                     torch.rand((2, 3, args.image_size, args.image_size)).to(args.device),
                                     torch.rand((2, 3, 257, 231)).to(args.device)]]

        pool_dim = 512
        self.pool_dim = pool_dim

        self.face_module = face_module(fusion_layer=4, pool_dim=pool_dim)
        self.body_module = body_module(fusion_layer=4, pool_dim=pool_dim)
        self.voice_module = voice_module(fusion_layer=4, pool_dim=pool_dim)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(768, 2048),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Dropout(0.5)
        )

        nb_modalities = 3

        self.fusion_layer = 4

        self.shared_resnet = shared_resnet(fusion_layer=4, pool_dim=pool_dim)

        pool_dim = 2 * pool_dim

        # self.fc = nn.Linear(pool_dim, 461, bias=False)
        self.l2norm = Normalize(2)

        self.nb_modalities = nb_modalities

        self.args = args
        self.batch_size = self.args.batch_size
        self.learning_rate = self.args.learning_rate

        # loading Embedding Aggregators
        if isinstance(args.embedding_aggregator.__target__, ListConfig):
            self.embedding_aggregator = []
            for i in range(len(args.embedding_aggregator.__target__)):
                __E_pkg__ = ".".join(args.embedding_aggregator.__target__[i].split(".")[:-1])
                __E_name__ = args.embedding_aggregator.__target__[i].split(".")[-1]
                __E__ = importlib.import_module(__E_pkg__).__getattribute__(__E_name__)
                if issubclass(__E__, pl.LightningModule):
                    embedding_aggregator = __E__.load_from_checkpoint(args.embedding_aggregator.args[i].checkpoint,
                                                                      args=OmegaConf.load(
                                                                          args.embedding_aggregator.args[
                                                                              i].cfg).experiments.model)
                    embedding_aggregator = embedding_aggregator.embedding_aggregator
                    embedding_aggregator.eval().requires_grad_(False)
                    if args.embedding_aggregator.args[i].grad[0] == "all":
                        embedding_aggregator.train().requires_grad_(True)
                    for layer in embedding_aggregator.named_modules():
                        if layer[0] in args.embedding_aggregator.args[i].grad:
                            layer[1].train().requires_grad_()
                    self.embedding_aggregator.append(embedding_aggregator)
                else:
                    self.embedding_aggregator.append(__E__(**args.embedding_aggregator.args[i]))
                    ckpt = args.embedding_aggregator.args[i].get("checkpoint", None)
                    if ckpt is not None:
                        try:
                            self.embedding_aggregator[i].load_state_dict(torch.load(ckpt)["state_dict"], strict=False)
                        except:
                            self.embedding_aggregator[i].load_state_dict(torch.load(ckpt), strict=False)

            self.embedding_aggregator = nn.ModuleList(self.embedding_aggregator)
        else:
            print("embedding_aggregator.__target__ should be a list")
            raise TypeError

        # loading Embedding Mixer
        __M_pkg__ = ".".join(args.embedding_mixer.__target__.split(".")[:-1])
        __M_name__ = args.embedding_mixer.__target__.split(".")[-1]
        __M__ = importlib.import_module(__M_pkg__).__getattribute__(__M_name__)
        self.embedding_mixer = __M__(**args.embedding_mixer)

        # loading Loss Function & Classifier
        __L_pkg__ = ".".join(args.loss_function.__target__.split(".")[:-1])
        __L_name__ = args.loss_function.__target__.split(".")[-1]
        __L__ = importlib.import_module(__L_pkg__).__getattribute__(__L_name__)
        self.loss_function = __L__(**args.loss_function)

        # loading Optimizer
        __O_pkg__ = ".".join(args.optimizer.__target__.split(".")[:-1])
        __O_name__ = args.optimizer.__target__.split(".")[-1]
        __O__ = importlib.import_module(__O_pkg__).__getattribute__(__O_name__)
        self.optim = __O__

        # loading Learning Rate Scheduler
        __S_pkg__ = ".".join(args.optimizer.lr_scheduler.__target__.split(".")[:-1])
        __S_name__ = args.optimizer.lr_scheduler.__target__.split(".")[-1]
        __S__ = importlib.import_module(__S_pkg__).__getattribute__(__S_name__)
        self.lr_sh = __S__

    def forward(self, X):

        X[0] = self.face_module(X[0])  # X[0] = (X_pool, feat)
        X[1] = self.body_module(X[1])  # X[1] = (X_pool, feat)
        X[2] = self.voice_module(X[2])

        f_embd = self.avg_pool(X[0])
        b_embd = self.avg_pool(X[1])
        v_embd = self.avg_pool(X[2])

        embed = torch.cat([f_embd, b_embd, v_embd], 1).squeeze()
        embed = self.fc(embed)
        return embed
        # if self.training:
        #     return x_pool, self.fc(feat)
        # else:
        #     return self.l2norm(x_pool), self.l2norm(feat)

    def configure_optimizers(self):
        opt = self.optim(self.parameters(), **self.args.optimizer.args)
        if self.args.optimizer.lr_scheduler.args.last_epoch is None:
            if self.args.optimizer.lr_scheduler.interval == "step":
                self.args.optimizer.lr_scheduler.args.last_epoch = self.global_step - 1
            else:
                self.args.optimizer.lr_scheduler.args.last_epoch = self.current_epoch - 1
        lr_sh = self.lr_sh(opt, **self.args.optimizer.lr_scheduler.args)
        return [opt], [{'interval': self.args.optimizer.lr_scheduler.interval, 'scheduler': lr_sh}]

    def training_step(self, batch, batch_idx):
        x, y = batch
        embeddings = self(x)
        loss = self.loss_function(embeddings, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, label, pathes = batch
        embeddings = self(x)
        embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings, pathes

    def validation_epoch_end(self, val_step_outputs: List[Any]) -> None:
        # test REID
        embeds = torch.cat([x[0] for x in val_step_outputs])
        pathes = list(np.concatenate([x[1] for x in val_step_outputs], 0))
        d_embeddings = {}
        if not len(pathes) == len(embeds):
            print(len(pathes), len(embeds))
            exit()
        for k, v in zip(pathes, embeds):
            try:
                key = k[:-10]
            except:
                key = k[0][:-10]
            d_embeddings[key] = torch.squeeze(v).cpu()
        y_score = []  # score for each sample
        y = []  # label for each sample
        with open(self.args.trial_path, 'r') as f:
            l_trial = f.readlines()
        for line in l_trial:
            trg, utt_a, utt_b = line.strip().split(' ')
            y.append(int(trg))
            y_score.append(torch.cosine_similarity(d_embeddings[utt_a[:-10]], d_embeddings[utt_b[:-10]], dim=0))
        y_score = torch.tensor(y_score)
        tuned_threshold, eer, fpr, fnr = tune_threshold_from_score(y_score, y, [1, 0.1])
        fnrs, fprs, thresholds = compute_error_rates(y_score, y)
        min_dcf, min_c_det_threshold = compute_min_dcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        try:
            self.log_dict({"EER": eer, "MinDCF": min_dcf}, on_step=False, on_epoch=True)
        except:
            print("Epoch {}: EER logging failed!".format(self.current_epoch))
        print("epoch {} step {}: EER reaches {}%".format(self.current_epoch, self.global_step, eer))

    def test_step(self, batch, batch_idx):
        self.count += 1
        if self.count == 113:
            print("113")
        x, label, pathes = batch
        union_embeddings = self(x)
        union_embeddings = torch.nn.functional.normalize(union_embeddings)
        return union_embeddings, pathes

    def test_epoch_end(self, outputs: List[Any]) -> None:
        # test REID
        result = ["epoch:{}".format(self.current_epoch),
                  "step:{}".format(self.global_step),
                  "ckpt:{}".format(self.args.checkpoint)]

        union_embeds = torch.cat([x[0] for x in outputs])
        pathes = list(np.concatenate([x[-1] for x in outputs], 0))
        d_embeddings = {}
        if not len(pathes) == len(union_embeds):
            print(len(pathes), len(union_embeds))
            exit()
        for k, v1 in zip(pathes, union_embeds):
            try:
                key = k[:-10]
            except:
                key = k[0][:-10]
            d_embeddings[key] = torch.squeeze(v1).cpu()

        y_score = []  # score for each sample
        y = []  # label for each sample
        with open(self.args.trial_path, 'r') as f:
            l_trial = f.readlines()
        for line in tqdm(l_trial, desc="cal_score"):
            trg, utt_a, utt_b = line.strip().split(' ')
            y.append(int(trg))
            y_score.append(torch.cosine_similarity(d_embeddings[utt_a[:-10]], d_embeddings[utt_b[:-10]], dim=0))
        y_score = torch.tensor(y_score)
        tuned_threshold, eer, fpr, fnr = tune_threshold_from_score(y_score, y, [1, 0.1])
        fnrs, fprs, thresholds = compute_error_rates(y_score, y)
        min_dcf, min_c_det_threshold = compute_min_dcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        self.log_dict({"EER": eer, "MinDCF": min_dcf}, on_step=False, on_epoch=True)
        print("Calculated by compute_min_cost:\nEER = {0}%\nminDCF = {1}\n".format(eer, min_dcf))
        print()
        result.append(
            "Epoch {} step {}:\nEER = {}%\nminDCF = {}".format(self.current_epoch, self.global_step, eer, min_dcf))
        with open("result.txt", 'w') as f:
            f.write("\n".join(result))


# endregion # <--- ML_MDA ---> #

class MFFN_DBFF(MFFN_BASE):
    def __init__(self, args: DictConfig):
        super(MFFN_DBFF, self).__init__(args)

        self.attxz_voice = nn.Conv1d(256, 512, 1)  # 256 * 80
        self.attxz_video = nn.Conv1d(512, 256, 1)  # 512 * 4 * 14 * 14
        self.attxz_expand = nn.Conv1d(512, 256, 1)
        self.attxz_bn = nn.BatchNorm1d(256)
        self.attzx_expand = nn.Conv3d(1024, 512, 1)
        self.attzx_bn = nn.BatchNorm3d(512)

        self.attx_fc = nn.Linear(1024, 512)
        self.atty_fc = nn.Identity()
        self.attz_fc = nn.Linear(2048, 512)

        self.attxy_cbp = CompactBilinearPooling(512, 512, 1024)
        self.attxz_cbp = CompactBilinearPooling(512, 512, 1024)
        self.attyz_cbp = CompactBilinearPooling(512, 512, 1024)
        print("Using DBFF")

    def forward(self, inpt, **kwargs) -> Any:
        # region    # <--- RawNet ---> #
        nb_samp = inpt[2].shape[0]
        len_seq = inpt[2].shape[1]
        x = self.voice_ln(inpt[2])
        x = x.view(nb_samp, 1, len_seq)
        x = F.max_pool1d(torch.abs(self.voice_first_conv(x)), 3)
        x = self.voice_first_bn(x)
        x = self.voice_lrelu_keras(x)
        x = self.voice_block0(x)  # b 128 2177
        x = self.voice_block1(x)  # 256 725
        x = self.voice_block2(x)  # 256 241
        x = self.voice_block3(x)  # 256 80
        x = self.voice_block4(x)  # 256 26

        z = self.video_conv1(inpt[1])
        z = self.video_bn1(z)
        z = self.video_relu(z)
        z = self.video_maxpool(z)
        z = self.video_layer1(z)  # 256 8 28 28
        z = self.video_layer2(z)  # 512 4 14 14

        # x:[b 256 80] y:[] z:[512 4 14 14]
        adj_z2x = self.attxz_video(F.adaptive_avg_pool3d(z, (80, 1, 1)).squeeze())  # 256 80
        adj_x2z = F.adaptive_avg_pool1d(self.attxz_voice(x), 4).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 14,
                                                                                                   14)  # 512 4 14 14
        stack_x = torch.cat((x, adj_z2x), dim=1)  # 256+256 80
        stack_z = torch.cat((z, adj_x2z), dim=1)  # 512+512 4 14 14
        x = self.attxz_expand(stack_x)
        z = self.attzx_expand(stack_z)

        x = self.voice_block5(x)  # 256 26
        x = self.voice_bn_before_gru(x)
        x = self.voice_lrelu_keras(x)
        x = x.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
        self.voice_gru.flatten_parameters()
        x, _ = self.voice_gru(x)  # 26 1024
        x = x[:, -1, :]  # 1024
        x = self.voice_fc1_gru(x)  # 1024
        # endregion # <--- RA ---> #

        # region    # <--- II-Net ---> #
        y = self.face_conv1(inpt[0])
        y = self.face_bn1(y)
        y = self.face_prelu(y)
        y = self.face_layer1(y)  # 64 56 56
        y = self.face_layer2(y)  # 64 28 28
        y = self.face_layer3(y)  # 128 14 14
        y = self.face_layer4(y)  # 512 7 7
        y = self.face_bn2(y)
        y = self.face_dropout(y)
        # endregion # <--- II-Net ---> #

        # region    # <--- Motion ---> #

        z = self.video_layer3(z)  # 1024 2 7 7
        z = self.video_layer4(z)  # 2048 1 4 4
        # endregion # <--- Motion ---> #

        embed_x = self.embedding_aggregator[2](x)
        embed_y = self.embedding_aggregator[0](y)
        embed_z = self.embedding_aggregator[1](z)

        embed = self.embedding_mixer([embed_x, embed_y, embed_z])
        return embed


# region    # <--- HAFM ---> #

class MFFN_HAFM(MFFN_BASE):
    def __init__(self, args: DictConfig):
        super(MFFN_HAFM, self).__init__(args)
        self.fcx = nn.Linear(1536, 1536)
        self.fcy = nn.Linear(1536, 1536)
        self.fcz = nn.Linear(1536, 1536)
        self.fcxyz = nn.Linear(4608, 1536)

        self.attx_fc = nn.Linear(1024, 512)
        self.atty_fc = nn.Identity()
        self.attz_fc = nn.Linear(2048, 512)

    def ca(self, inpt_x, inpt_y) -> torch.Tensor:
        x = torch.matmul(inpt_x.T, inpt_y)
        x = F.tanh(x)
        x = F.softmax(x, dim=-1)
        x = torch.matmul(inpt_x, x)
        return x

    def forward(self, inpt, **kwargs) -> Any:
        # region    # <--- RawNet ---> #
        nb_samp = inpt[2].shape[0]
        len_seq = inpt[2].shape[1]
        x = self.voice_ln(inpt[2])
        x = x.view(nb_samp, 1, len_seq)
        x = F.max_pool1d(torch.abs(self.voice_first_conv(x)), 3)
        x = self.voice_first_bn(x)
        x = self.voice_lrelu_keras(x)
        x = self.voice_block0(x)
        x = self.voice_block1(x)
        x = self.voice_block2(x)
        x = self.voice_block3(x)
        x = self.voice_block4(x)
        x = self.voice_block5(x)
        x = self.voice_bn_before_gru(x)
        x = self.voice_lrelu_keras(x)
        x = x.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
        self.voice_gru.flatten_parameters()
        x, _ = self.voice_gru(x)
        x = x[:, -1, :]
        x = self.voice_fc1_gru(x)
        # endregion # <--- RA ---> #

        # region    # <--- II-Net ---> #
        y = self.face_conv1(inpt[0])
        y = self.face_bn1(y)
        y = self.face_prelu(y)
        y = self.face_layer1(y)
        y = self.face_layer2(y)
        y = self.face_layer3(y)
        y = self.face_layer4(y)
        y = self.face_bn2(y)
        y = self.face_dropout(y)
        # endregion # <--- II-Net ---> #

        # region    # <--- Motion ---> #
        z = self.video_conv1(inpt[1])
        z = self.video_bn1(z)
        z = self.video_relu(z)
        z = self.video_maxpool(z)

        z = self.video_layer1(z)
        z = self.video_layer2(z)
        z = self.video_layer3(z)
        z = self.video_layer4(z)
        # endregion # <--- Motion ---> #

        embed_x = self.embedding_aggregator[2](x)
        embed_y = self.embedding_aggregator[0](y)
        embed_z = self.embedding_aggregator[1](z)

        embed_x = self.attx_fc(embed_x)
        embed_y = self.atty_fc(embed_y)
        embed_z = self.attz_fc(embed_z)

        embed_x2 = torch.cat([self.ca(embed_x, embed_x), self.ca(embed_x, embed_y), self.ca(embed_x, embed_z)], dim=-1)
        embed_y2 = torch.cat([self.ca(embed_y, embed_y), self.ca(embed_y, embed_z), self.ca(embed_y, embed_x)], dim=-1)
        embed_z2 = torch.cat([self.ca(embed_z, embed_z), self.ca(embed_z, embed_x), self.ca(embed_z, embed_y)], dim=-1)
        embed_xyz = torch.cat([embed_x2, embed_y2, embed_z2], dim=-1)

        ux = F.tanh(self.fcx(embed_x2))
        uy = F.tanh(self.fcy(embed_y2))
        uz = F.tanh(self.fcz(embed_z2))
        u = F.tanh(self.fcxyz(embed_xyz))

        ax = F.softmax(torch.matmul(ux.T, u), dim=-1)
        ay = F.softmax(torch.matmul(uy.T, u), dim=-1)
        az = F.softmax(torch.matmul(uz.T, u), dim=-1)

        embed = torch.matmul(ux, ax) + torch.matmul(uy, ay) + torch.matmul(uz, az)

        embed = self.embedding_mixer([embed])
        return embed


# endregion # <--- HAFM ---> #

# endregion # <--- Compare ---> #

pl.seed_everything(970614, True)


def train(cfg: DictConfig):
    data_module = pl_models.MMClassificationData(cfg.model)
    model = MFFN_HAFM(cfg.model)
    callbacks = trainer_manager.get_callbacks(cfg.callbacks)
    loggers = trainer_manager.get_loggers(cfg.loggers)
    trainer = pl.Trainer(**cfg.trainer, **{"callbacks": callbacks, "logger": loggers})
    trainer.fit(model, datamodule=data_module)


def test(cfg: DictConfig):
    data_module = pl_models.MMClassificationData(OmegaConf.load(cfg.model.config).experiments.model)
    model = MMClassification_XZ1.load_from_checkpoint(cfg.model.checkpoint,
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


if __name__ == '__main__':
    model = MFFN_Face_Audio(OmegaConf.load("/home/yjc/PythonProject/PLIntegration/PLIntegration/conf/"
                                           "experiments/multi_modal_classification/mma.yaml").model)
    model.to("cuda")
    model_summary = summarize(model, max_depth=1)
    summary_data = model_summary._get_summary_data()
    total_parameters = model_summary.total_parameters
    trainable_parameters = model_summary.trainable_parameters
    model_size = model_summary.model_size
    summary_table = _format_summary_table(total_parameters, trainable_parameters, model_size, *summary_data)
    print("\n" + summary_table, end="")
