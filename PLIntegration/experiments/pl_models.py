#!usr/bin/env python
# -*- coding:utf-8 _*-
import importlib
from abc import ABC
from glob import glob
from typing import Any, List, Optional
from torch import nn
import joblib
import pytorch_lightning as pl
import torch.optim.sgd
from omegaconf import DictConfig, OmegaConf, ListConfig
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.nn.functional import normalize
from tqdm import tqdm

from PLIntegration.experiments.utils.dataloader_manager import get_dataloaders
from PLIntegration.metrics.subjective import sub_mm2
from PLIntegration.metrics.verification import *


class FaceClassification(pl.LightningModule):
    def __init__(self, args: DictConfig):
        super().__init__()
        self.example_input_array = [
            torch.rand((2, 3, args.image_size, args.image_size)).to(args.device)]
        self.args = args
        self.batch_size = self.args.batch_size
        self.learning_rate = self.args.learning_rate

        # loading Feature Extractor
        __F_pkg__ = ".".join(args.feature_extractor.__target__.split(".")[:-1])
        __F_name__ = args.feature_extractor.__target__.split(".")[-1]
        __F__ = importlib.import_module(__F_pkg__).__getattribute__(__F_name__)
        self.feature_extractor = __F__(**args.feature_extractor)

        # loading Embedding Aggregator
        __E_pkg__ = ".".join(args.embedding_aggregator.__target__.split(".")[:-1])
        __E_name__ = args.embedding_aggregator.__target__.split(".")[-1]
        __E__ = importlib.import_module(__E_pkg__).__getattribute__(__E_name__)
        self.embedding_aggregator = __E__(**args.embedding_aggregator)

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

    def forward(self, inpt, **kwargs) -> Any:
        features = self.feature_extractor(inpt)
        embeddings = self.embedding_aggregator(features)
        return embeddings

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
            key = "/".join(k.split("/")[-2:])[:-4]
            d_embeddings[key] = torch.squeeze(v).cpu()
        y_score = []  # score for each sample
        y = []  # label for each sample
        with open(self.args.trial_path, 'r') as f:
            l_trial = f.readlines()
        for line in l_trial:
            trg, utt_a, utt_b = line.strip().split(' ')
            y.append(int(trg))
            y_score.append(torch.cosine_similarity(d_embeddings[utt_a[:-4]], d_embeddings[utt_b[:-4]], dim=0))
        y_score = torch.tensor(y_score)
        tuned_threshold, eer, fpr, fnr = tune_threshold_from_score(y_score, y, [1, 0.1])
        fnrs, fprs, thresholds = compute_error_rates(y_score, y)
        min_dcf, min_c_det_threshold = compute_min_dcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        print("Calculated by compute_min_cost:\nEER = {0}%\nminDCF = {1}\n".format(eer, min_dcf))
        print(r"REID: ${:.2f}\\{:.2f}$".format(eer, min_dcf))
        print()
        try:
            self.log_dict({"EER": eer, "MinDCF": min_dcf}, on_step=False, on_epoch=True)
        except:
            print("Epoch {}: EER logging failed!".format(self.current_epoch))
        print("Epoch {}: EER reaches {}%".format(self.current_epoch, eer))

    def test_step(self, batch, batch_idx):
        x, label, pathes = batch
        embeddings = self(x)
        embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings, pathes

    def test_epoch_end(self, outputs: List[Any]) -> None:
        # test REID
        result = ["epoch:{}".format(self.current_epoch),
                  "step:{}".format(self.global_step),
                  "ckpt:{}".format(self.args.checkpoint)]

        embeds = torch.cat([x[0] for x in outputs])
        pathes = list(np.concatenate([x[1] for x in outputs], 0))
        d_embeddings = {}
        if not len(pathes) == len(embeds):
            print(len(pathes), len(embeds))
            exit()
        for k, v in zip(pathes, embeds):
            key = "/".join(k.split("/")[-2:])[:-4]
            d_embeddings[key] = torch.squeeze(v).cpu()
        joblib.dump(d_embeddings, "embd_face_train.pkl", protocol=4)
        y_score = []  # score for each sample
        y = []  # label for each sample
        with open(self.args.trial_path, 'r') as f:
            l_trial = f.readlines()
        for line in l_trial:
            trg, utt_a, utt_b = line.strip().split(' ')
            y.append(int(trg))
            y_score.append(torch.cosine_similarity(d_embeddings[utt_a[:-4]], d_embeddings[utt_b[:-4]], dim=0))
        y_score = torch.tensor(y_score)
        tuned_threshold, eer, fpr, fnr = tune_threshold_from_score(y_score, y, [1, 0.1])
        fnrs, fprs, thresholds = compute_error_rates(y_score, y)
        min_dcf, min_c_det_threshold = compute_min_dcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        print("Calculated by compute_min_cost:\nEER = {0}%\nminDCF = {1}\n".format(eer, min_dcf))
        print(r"REID: ${:.2f}\\{:.2f}$".format(eer, min_dcf))
        print()
        result.append("Calculated by compute_min_cost:\nEER = {0}%\nminDCF = {1}".format(eer, min_dcf))
        self.log_dict({"EER": eer, "MinDCF": min_dcf}, on_step=False, on_epoch=True)
        # test lfws
        l_fprs, l_tprs, names = [], [], []
        for bin_file in glob("/home/yjc/PythonProject/re-identity-with-vision/datasets/faceDB/faces_eval/*.bin"):
            data_set = load_bin(bin_file, (self.args.image_size, self.args.image_size))
            name = bin_file.split('/')[-1].split('.')[0]
            val, val_std, far, acc, std, tprs, fprs = test(data_set, self,
                                                           batch_size=32,
                                                           nfolds=10,
                                                           device=self.args.device)
            l_fprs.append(fprs)
            l_tprs.append(tprs)
            names.append(name)
            print("{}: val={:.4f}%+-{:.4f}% @ far={}".format(name, val * 100, val_std * 100, far))
            print("{}: acc={:.4f}%+-{:.4f}%".format(name, acc * 100, std * 100))
            print(
                r"{}: ${:.2f}\pm{:.2f}\\{:.2f}\pm{:.2f}$".format(name, val * 100, val_std * 100, acc * 100, std * 100))
            print()
            self.log_dict({name + '_val': val, name + '_val_std': val_std, name + "_far": far}, on_step=False,
                          on_epoch=True)
            self.log_dict({name + '_acc': acc, name + '_std': std}, on_step=False, on_epoch=True)
            result.append("{}: val={:.4f}%+-{:.4f}% @ far={}".format(name, val * 100, val_std * 100, far))
            result.append("{}: acc={:.4f}%+-{:.4f}%".format(name, acc * 100, std * 100))
        # log_path = os.path.join(self.logger.save_dir, self.logger.name)
        with open("result.txt", 'w') as f:
            f.write("\n".join(result))
        joblib.dump((l_tprs, l_fprs, names), "roc.pkl", protocol=4)


class FaceClassificationData(pl.LightningDataModule):
    def __init__(self, args: DictConfig):
        super().__init__()
        self.args = args

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        loaders = get_dataloaders(self.args.train_ds)
        if len(loaders) == 1:
            return loaders[0]
        else:
            return loaders

    def val_dataloader(self) -> EVAL_DATALOADERS:
        loaders = get_dataloaders(self.args.validation_ds)
        if len(loaders) == 1:
            return loaders[0]
        else:
            return loaders

    def test_dataloader(self) -> EVAL_DATALOADERS:
        loaders = get_dataloaders(self.args.test_ds)
        if len(loaders) == 1:
            return loaders[0]
        else:
            return loaders

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        loaders = get_dataloaders(self.args.test_ds)
        if len(loaders) == 1:
            return loaders[0]
        else:
            return loaders

    def teardown(self, stage: Optional[str] = None) -> None:
        pass


class VideoClassification(pl.LightningModule):
    def __init__(self, args: DictConfig):
        super().__init__()
        self.example_input_array = [
            torch.rand((2, 3, 16, args.image_size, args.image_size)).to(args.device)]
        self.args = args
        self.batch_size = self.args.batch_size
        self.learning_rate = self.args.learning_rate

        # loading Feature Extractor
        __F_pkg__ = ".".join(args.feature_extractor.__target__.split(".")[:-1])
        __F_name__ = args.feature_extractor.__target__.split(".")[-1]
        __F__ = importlib.import_module(__F_pkg__).__getattribute__(__F_name__)
        self.feature_extractor = __F__(**args.feature_extractor)

        # loading Embedding Aggregator
        __E_pkg__ = ".".join(args.embedding_aggregator.__target__.split(".")[:-1])
        __E_name__ = args.embedding_aggregator.__target__.split(".")[-1]
        __E__ = importlib.import_module(__E_pkg__).__getattribute__(__E_name__)
        self.embedding_aggregator = __E__(**args.embedding_aggregator)

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

    def forward(self, inpt, **kwargs) -> Any:
        features = self.feature_extractor(inpt)
        embeddings = self.embedding_aggregator(features)
        return embeddings

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
        x, label, pathes = batch
        embeddings = self(x)
        embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings, pathes

    def test_epoch_end(self, outputs: List[Any]) -> None:
        # test REID
        result = ["epoch:{}".format(self.current_epoch),
                  "step:{}".format(self.global_step),
                  "ckpt:{}".format(self.args.checkpoint)]

        embeds = torch.cat([x[0] for x in outputs])
        pathes = list(np.concatenate([x[1] for x in outputs], 0))
        d_embeddings = {}
        if not len(pathes) == len(embeds):
            print(len(pathes), len(embeds))
            exit()
        for k, v in zip(pathes, embeds):
            key = "/".join(k.split("/")[-2:])[:-4]
            d_embeddings[key] = torch.squeeze(v).cpu()
        joblib.dump(d_embeddings, "embd_face_train.pkl", protocol=4)
        y_score = []  # score for each sample
        y = []  # label for each sample
        with open(self.args.trial_path, 'r') as f:
            l_trial = f.readlines()
        for line in l_trial:
            trg, utt_a, utt_b = line.strip().split(' ')
            y.append(int(trg))
            y_score.append(torch.cosine_similarity(d_embeddings[utt_a[:-4]], d_embeddings[utt_b[:-4]], dim=0))
        y_score = torch.tensor(y_score)
        tuned_threshold, eer, fpr, fnr = tune_threshold_from_score(y_score, y, [1, 0.1])
        fnrs, fprs, thresholds = compute_error_rates(y_score, y)
        min_dcf, min_c_det_threshold = compute_min_dcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        print("Calculated by compute_min_cost:\nEER = {0}%\nminDCF = {1}\n".format(eer, min_dcf))
        print(r"REID: ${:.2f}\\{:.2f}$".format(eer, min_dcf))
        print()
        result.append("Calculated by compute_min_cost:\nEER = {0}%\nminDCF = {1}".format(eer, min_dcf))
        self.log_dict({"EER": eer, "MinDCF": min_dcf}, on_step=False, on_epoch=True)
        with open("result.txt", 'w') as f:
            f.write("\n".join(result))


class VideoClassificationData(pl.LightningDataModule):
    def __init__(self, args: DictConfig):
        super().__init__()
        self.args = args

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        loaders = get_dataloaders(self.args.train_ds)
        if len(loaders) == 1:
            return loaders[0]
        else:
            return loaders

    def val_dataloader(self) -> EVAL_DATALOADERS:
        loaders = get_dataloaders(self.args.validation_ds)
        if len(loaders) == 1:
            return loaders[0]
        else:
            return loaders

    def test_dataloader(self) -> EVAL_DATALOADERS:
        loaders = get_dataloaders(self.args.test_ds)
        if len(loaders) == 1:
            return loaders[0]
        else:
            return loaders

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        loaders = get_dataloaders(self.args.test_ds)
        if len(loaders) == 1:
            return loaders[0]
        else:
            return loaders

    def teardown(self, stage: Optional[str] = None) -> None:
        pass


class MMClassification(pl.LightningModule):
    def __init__(self, args: DictConfig):
        super().__init__()
        if args.get("nb_samp", False):
            self.example_input_array = [[torch.rand((2, 3, args.image_size, args.image_size)).to(args.device),
                                         torch.rand((2, 3, 16, args.image_size, args.image_size)).to(args.device),
                                         torch.rand((2, args.nb_samp)).to(args.device)]]
        else:
            self.example_input_array = [[torch.rand((2, 3, args.image_size, args.image_size)).to(args.device),
                                         torch.rand((2, 3, 16, args.image_size, args.image_size)).to(args.device)]]
        self.args = args
        self.batch_size = self.args.batch_size
        self.learning_rate = self.args.learning_rate

        # loading Feature Extractors
        if isinstance(args.feature_extractor.__target__, ListConfig):
            self.feature_extractor = []
            for i in range(len(args.feature_extractor.__target__)):
                __F_pkg__ = ".".join(args.feature_extractor.__target__[i].split(".")[:-1])
                __F_name__ = args.feature_extractor.__target__[i].split(".")[-1]
                __F__ = importlib.import_module(__F_pkg__).__getattribute__(__F_name__)
                if issubclass(__F__, pl.LightningModule):
                    feature_extractor = __F__.load_from_checkpoint(args.feature_extractor.args[i].checkpoint,
                                                                   args=OmegaConf.load(args.feature_extractor.args[
                                                                                           i].cfg).experiments.model)
                    feature_extractor = feature_extractor.feature_extractor
                    for layer in feature_extractor.named_modules():
                        if "all" in args.feature_extractor.args[i].grad:
                            layer[1].train().requires_grad_()
                        if layer[0] in args.feature_extractor.args[i].grad:
                            layer[1].train().requires_grad_()
                        elif "layer" in layer[0]:
                            if ".".join(layer[0].split(".")[:-1]) in args.feature_extractor.args[i].grad:
                                layer[1].train().requires_grad_()
                            if ".".join(layer[0].split(".")[:-2]) in args.feature_extractor.args[i].grad:
                                layer[1].train().requires_grad_()
                        else:
                            layer[1].eval().requires_grad_(False)
                    self.feature_extractor.append(feature_extractor)
                else:
                    self.feature_extractor.append(__F__(**args.feature_extractor.args[i]))
                    ckpt = args.feature_extractor.args[i].get("checkpoint", None)
                    if ckpt is not None:
                        try:
                            self.feature_extractor[i].load_state_dict(torch.load(ckpt)["state_dict"], strict=False)
                        except:
                            self.feature_extractor[i].load_state_dict(torch.load(ckpt), strict=False)
            self.feature_extractor = nn.ModuleList(self.feature_extractor)
        else:
            print("feature_extractor.__target__ should be a list")
            raise TypeError

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

    def forward(self, inpt, **kwargs) -> Any:
        embeddings = []
        for i in range(len(self.feature_extractor)):
            features = self.feature_extractor[i](inpt[i])
            embeddings.append(self.embedding_aggregator[i](features))
        embeddings = self.embedding_mixer(embeddings)
        return embeddings

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
        x, label, pathes = batch
        union_embeddings = self(x)
        union_embeddings = torch.nn.functional.normalize(union_embeddings)
        face_embeddings = self.embedding_aggregator[0](self.feature_extractor[0](x[0]))
        face_embeddings = torch.nn.functional.normalize(face_embeddings)
        video_embeddings = self.embedding_aggregator[1](self.feature_extractor[1](x[1]))
        video_embeddings = torch.nn.functional.normalize(video_embeddings)
        if len(self.feature_extractor) == 3:
            audio_embeddings = self.embedding_aggregator[2](self.feature_extractor[2](x[2]))
            audio_embeddings = torch.nn.functional.normalize(audio_embeddings)
            return union_embeddings, face_embeddings, video_embeddings, audio_embeddings, pathes
        return union_embeddings, face_embeddings, video_embeddings, pathes

    def test_epoch_end(self, outputs: List[Any]) -> None:
        # test REID
        result = ["epoch:{}".format(self.current_epoch),
                  "step:{}".format(self.global_step),
                  "ckpt:{}".format(self.args.checkpoint)]

        union_embeds = torch.cat([x[0] for x in outputs])
        face_embeds = torch.cat([x[1] for x in outputs])
        video_embeds = torch.cat([x[2] for x in outputs])
        if len(outputs[0]) == 5:
            audio_embeds = torch.cat([x[3] for x in outputs])
        pathes = list(np.concatenate([x[-1] for x in outputs], 0))
        d_embeddings = {}
        if not len(pathes) == len(union_embeds):
            print(len(pathes), len(union_embeds))
            exit()
        if len(outputs[0]) != 5:
            for k, v1, v2, v3 in zip(pathes, union_embeds, face_embeds, video_embeds):
                try:
                    key = k[:-10]
                except:
                    key = k[0][:-10]
                d_embeddings[key] = (torch.squeeze(v1).cpu(), torch.squeeze(v2).cpu(), torch.squeeze(v3).cpu())
        else:
            for k, v1, v2, v3, v4 in zip(pathes, union_embeds, face_embeds, video_embeds, audio_embeds):
                try:
                    key = k[:-10]
                except:
                    key = k[0][:-10]
                d_embeddings[key] = (
                    torch.squeeze(v1).cpu(), torch.squeeze(v2).cpu(), torch.squeeze(v3).cpu(), torch.squeeze(v4).cpu())
        if len(outputs[0]) == 5:
            y_score = [[], [], [], []]  # score for each sample
        else:
            y_score = [[], [], []]
        y = []  # label for each sample
        with open(self.args.trial_path, 'r') as f:
            l_trial = f.readlines()
        for line in tqdm(l_trial, desc="cal_score"):
            trg, utt_a, utt_b = line.strip().split(' ')
            y.append(int(trg))
            y_score[0].append(
                torch.cosine_similarity(d_embeddings[utt_a[:-10]][0], d_embeddings[utt_b[:-10]][0], dim=0))
            y_score[1].append(
                torch.cosine_similarity(d_embeddings[utt_a[:-10]][1], d_embeddings[utt_b[:-10]][1], dim=0))
            y_score[2].append(
                torch.cosine_similarity(d_embeddings[utt_a[:-10]][2], d_embeddings[utt_b[:-10]][2], dim=0))
            if len(outputs[0]) == 5:
                y_score[3].append(
                    torch.cosine_similarity(d_embeddings[utt_a[:-10]][3], d_embeddings[utt_b[:-10]][3], dim=0))
        y_score = torch.tensor(y_score)
        tuned_threshold, eer, fpr, fnr = tune_threshold_from_score(y_score[0], y, [1, 0.1])
        fnrs, fprs, thresholds = compute_error_rates(y_score[0], y)
        min_dcf, min_c_det_threshold = compute_min_dcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        self.log_dict({"EER": eer, "MinDCF": min_dcf}, on_step=False, on_epoch=True)
        print("Calculated by compute_min_cost:\nEER = {0}%\nminDCF = {1}\n".format(eer, min_dcf))
        print()
        result.append(
            "Epoch {} step {}:\nEER = {}%\nminDCF = {}".format(self.current_epoch, self.global_step, eer, min_dcf))

        tuned_threshold, eer, fpr, fnr = tune_threshold_from_score(y_score[1], y, [1, 0.1])
        fnrs, fprs, thresholds = compute_error_rates(y_score[1], y)
        min_dcf, min_c_det_threshold1 = compute_min_dcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        self.log_dict({"EER_face": eer, "MinDCF_face": min_dcf}, on_step=False, on_epoch=True)
        print("Calculated by compute_min_cost:\nEER_face = {0}%\nminDCF_face = {1}\n".format(eer, min_dcf))
        print()
        result.append(
            "Epoch {} step {}:\nEER_face = {}%\nminDCF_face = {}".format(self.current_epoch, self.global_step, eer,
                                                                         min_dcf))

        tuned_threshold, eer, fpr, fnr = tune_threshold_from_score(y_score[2], y, [1, 0.1])
        fnrs, fprs, thresholds = compute_error_rates(y_score[2], y)
        min_dcf, min_c_det_threshold = compute_min_dcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        self.log_dict({"EER_video": eer, "MinDCF_video": min_dcf}, on_step=False, on_epoch=True)
        print("Calculated by compute_min_cost:\nEER_video = {0}%\nminDCF_video = {1}\n".format(eer, min_dcf))
        print()
        result.append(
            "Epoch {} step {}:\nEER_video = {}%\nminDCF_video = {}".format(self.current_epoch, self.global_step, eer,
                                                                           min_dcf))

        if len(outputs[0]) == 5:
            tuned_threshold, eer, fpr, fnr = tune_threshold_from_score(y_score[3], y, [1, 0.1])
            fnrs, fprs, thresholds = compute_error_rates(y_score[3], y)
            min_dcf, min_c_det_threshold = compute_min_dcf(fnrs, fprs, thresholds, 0.05, 1, 1)
            self.log_dict({"EER_audio": eer, "MinDCF_audio": min_dcf}, on_step=False, on_epoch=True)
            print("Calculated by compute_min_cost:\nEER_audio = {0}%\nMinDCF_audio = {1}\n".format(eer, min_dcf))
            print()
            result.append(
                "Epoch {} step {}:\nEER_audio = {}%\nMinDCF_audio = {}".format(self.current_epoch, self.global_step,
                                                                               eer,
                                                                               min_dcf))
        with open("result.txt", 'w') as f:
            f.write("\n".join(result))


class MMClassificationData(pl.LightningDataModule):
    def __init__(self, args: DictConfig):
        super().__init__()
        self.args = args

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        loaders = get_dataloaders(self.args.train_ds)
        if len(loaders) == 1:
            return loaders[0]
        else:
            return loaders

    def val_dataloader(self) -> EVAL_DATALOADERS:
        loaders = get_dataloaders(self.args.validation_ds)
        if len(loaders) == 1:
            return loaders[0]
        else:
            return loaders

    def test_dataloader(self) -> EVAL_DATALOADERS:
        loaders = get_dataloaders(self.args.test_ds)
        if len(loaders) == 1:
            return loaders[0]
        else:
            return loaders

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        loaders = get_dataloaders(self.args.test_ds)
        if len(loaders) == 1:
            return loaders[0]
        else:
            return loaders

    def teardown(self, stage: Optional[str] = None) -> None:
        pass
