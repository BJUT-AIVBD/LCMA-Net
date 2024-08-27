#!usr/bin/env python
# -*- coding:utf-8 _*-
import os.path as osp

import joblib
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
import matplotlib.pyplot as plt
from PLIntegration.experiments.mma_reid import MFFN_DBFF, MFFN_Face_Audio, MFFN_HAFM, MFFN_ICODE1, ML_MDA, \
    MMClassification_XY1, \
    MMClassification_XY2, \
    MMClassification_XY3, \
    MMClassification_XY4, MMClassification_XZ1, MMClassification_YZ1, MMClassification_XYZ1, MMClassification_XY_XZ1, \
    MMClassification_XZ_YZ1, MMClassification_XY_YZ1, MFFN_MLB1, MFFN_CBP1, MFFN_attVALD1, MFFN_Transformer
from PLIntegration.experiments.pl_models import FaceClassification, VideoClassification, MMClassification
from PLIntegration.datasets.image import ManifestBase
from PLIntegration.datasets.video import MotionDataset
from PLIntegration.datasets.multi_modal import UnionDataset, UnionDataset3, UnionDataset3_MLMA, UnionDataset3_MLMDA
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from torch.utils.data import DataLoader
import torch.nn.functional as F

from PLIntegration.networks._2d.rawnet import RawNet_NL_GRU
from PLIntegration.networks._2d.seqface import SeqFace21
from thop import profile
from thop import clever_format


def get_face_embds(cfg: str, ckpt: str):
    transformers = Compose([Resize(112, 112),
                            ToTensor(),
                            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    model = FaceClassification.load_from_checkpoint(ckpt,
                                                    args=OmegaConf.load(cfg).experiments.model)
    dataset = ManifestBase("/home/yjc/PythonProject/PLIntegration/data/reid_face_all_manifest.json", transformers)
    dataloader = DataLoader(dataset,
                            batch_size=64,
                            num_workers=8,
                            pin_memory=True)
    model.cuda()
    model.eval()

    embeds_list, label_list = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            img = batch[0].to("cuda")
            label_list.append(batch[1])
            embeds_list.append(F.normalize(model(img)).to("cpu"))
        torch.save([torch.concat(embeds_list), torch.concat(label_list)],
                   ckpt.split(".ck")[0] + ".embds")
        # print("{} embeds saved".format(torch.concat(embeds_list).shape))


def get_face_embds2(cfg: str, ckpt: str):
    model = FaceClassification.load_from_checkpoint(ckpt,
                                                    args=OmegaConf.load(cfg).experiments.model)
    dataset = UnionDataset(face_dir="/data/REID/faces",
                           video_dir="/data/REID/frames",
                           image_size=112,
                           is_train=False,
                           train_split=587)
    dataloader = DataLoader(dataset,
                            batch_size=8,
                            num_workers=4,
                            pin_memory=True)
    model.cuda()
    model.eval()

    embeds_list, label_list = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            img = batch[0][0].to("cuda")
            label_list.append(batch[1])
            embeds_list.append(F.normalize(model(img)).to("cpu"))
        torch.save([torch.concat(embeds_list), torch.concat(label_list)],
                   ckpt.split(".ck")[0] + ".embds2")
        # print("{} embeds saved".format(torch.concat(embeds_list).shape))


def get_face_embds3(cfg: str, ckpt: str):
    model = FaceClassification.load_from_checkpoint(ckpt,
                                                    args=OmegaConf.load(cfg).experiments.model)
    dataset = UnionDataset3(face_dir="/data/REID/faces",
                            video_dir="/data/REID/frames",
                            audio_dir="/data/REID/audios",
                            image_size=112,
                            is_train=False,
                            train_split=587)
    dataloader = DataLoader(dataset,
                            batch_size=8,
                            num_workers=4,
                            pin_memory=True)
    model.cuda()
    model.eval()

    embeds_list, label_list = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            img = batch[0][0].to("cuda")
            label_list.append(batch[1])
            embeds_list.append(F.normalize(model(img)).to("cpu"))
        torch.save([torch.concat(embeds_list), torch.concat(label_list)],
                   ckpt.split(".ck")[0] + ".embds3")
        # print("{} embeds saved".format(torch.concat(embeds_list).shape))


def get_seqface_embds(cfg: str, ckpt: str):
    transformers = Compose([Resize(128),
                            ToTensor(),
                            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    model = SeqFace21()
    dataset = ManifestBase("/home/yjc/PythonProject/PLIntegration/data/reid_face_all_manifest.json", transformers)
    dataloader = DataLoader(dataset,
                            batch_size=64,
                            num_workers=8,
                            pin_memory=True)
    model.cuda()
    model.eval()

    embeds_list, label_list = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            img = batch[0].to("cuda")
            label_list.append(batch[1])
            embeds_list.append(F.normalize(model(img)).to("cpu"))
        torch.save([torch.concat(embeds_list), torch.concat(label_list)],
                   ckpt.split(".pt")[0] + ".embds")
        # print("{} embeds saved".format(torch.concat(embeds_list).shape))


def get_seqface_embds2(cfg: str, ckpt: str):
    model = SeqFace21()
    dataset = UnionDataset(face_dir="/data/REID/faces",
                           video_dir="/data/REID/frames",
                           image_size=128,
                           is_train=False,
                           train_split=587)
    dataloader = DataLoader(dataset,
                            batch_size=8,
                            num_workers=4,
                            pin_memory=True)
    model.cuda()
    model.eval()

    embeds_list, label_list = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            img = batch[0][0].to("cuda")
            label_list.append(batch[1])
            embeds_list.append(F.normalize(model(img)).to("cpu"))
        torch.save([torch.concat(embeds_list), torch.concat(label_list)],
                   ckpt.split(".pt")[0] + ".embds2")
        # print("{} embeds saved".format(torch.concat(embeds_list).shape))


def get_seqface_embds3(cfg: str, ckpt: str):
    model = SeqFace21()
    dataset = UnionDataset3(face_dir="/data/REID/faces",
                            video_dir="/data/REID/frames",
                            audio_dir="/data/REID/audios",
                            image_size=128,
                            is_train=False,
                            train_split=587)
    dataloader = DataLoader(dataset,
                            batch_size=8,
                            num_workers=4,
                            pin_memory=True)
    model.cuda()
    model.eval()

    embeds_list, label_list = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            img = batch[0][0].to("cuda")
            label_list.append(batch[1])
            embeds_list.append(F.normalize(model(img)).to("cpu"))
        torch.save([torch.concat(embeds_list), torch.concat(label_list)],
                   ckpt.split(".pt")[0] + ".embds3")
        # print("{} embeds saved".format(torch.concat(embeds_list).shape))


def get_video_embds(cfg: str, ckpt: str):
    transformers = Compose([Resize(112, 112),
                            ToTensor()])
    model = VideoClassification.load_from_checkpoint(ckpt,
                                                     args=OmegaConf.load(cfg).experiments.model)
    dataset = MotionDataset("/home/yjc/PythonProject/re-identity-with-vision/datasets/motionDB/REID/frames",
                            transformer=transformers,
                            is_train=False)
    dataloader = DataLoader(dataset,
                            batch_size=8,
                            num_workers=0,
                            pin_memory=True)
    model.cuda()
    model.eval()

    embeds_list, label_list = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            img = batch[0].to("cuda")
            label_list.append(batch[1])
            embeds_list.append(F.normalize(model(img)).to("cpu"))
        torch.save([torch.concat(embeds_list), torch.concat(label_list)],
                   ckpt.split(".ck")[0] + ".embds")
        # print("{} embeds saved".format(torch.concat(embeds_list).shape))


def get_video_embds2(cfg: str, ckpt: str):
    model = VideoClassification.load_from_checkpoint(ckpt,
                                                     args=OmegaConf.load(cfg).experiments.model)
    dataset = UnionDataset(face_dir="/data/REID/faces",
                           video_dir="/data/REID/frames",
                           image_size=112,
                           is_train=False,
                           train_split=587)
    dataloader = DataLoader(dataset,
                            batch_size=8,
                            num_workers=4,
                            pin_memory=True)
    model.cuda()
    model.eval()

    embeds_list, label_list = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            img = batch[0][1].to("cuda")
            label_list.append(batch[1])
            embeds_list.append(F.normalize(model(img)).to("cpu"))
        torch.save([torch.concat(embeds_list), torch.concat(label_list)],
                   ckpt.split(".ck")[0] + ".embds2")
        # print("{} embeds saved".format(torch.concat(embeds_list).shape))


def get_video_embds3(cfg: str, ckpt: str):
    model = VideoClassification.load_from_checkpoint(ckpt,
                                                     args=OmegaConf.load(cfg).experiments.model)
    dataset = UnionDataset3(face_dir="/data/REID/faces",
                            video_dir="/data/REID/frames",
                            audio_dir="/data/REID/audios",
                            image_size=112,
                            is_train=False,
                            train_split=587)
    dataloader = DataLoader(dataset,
                            batch_size=8,
                            num_workers=4,
                            pin_memory=True)
    model.cuda()
    model.eval()

    embeds_list, label_list = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            img = batch[0][1].to("cuda")
            label_list.append(batch[1])
            embeds_list.append(F.normalize(model(img)).to("cpu"))
        torch.save([torch.concat(embeds_list), torch.concat(label_list)],
                   ckpt.split(".ck")[0] + ".embds3")
        # print("{} embeds saved".format(torch.concat(embeds_list).shape))


def get_mm2_embds(cfg: str, ckpt: str):
    model = MMClassification.load_from_checkpoint(ckpt,
                                                  args=OmegaConf.load(cfg).experiments.model)
    dataset = UnionDataset(face_dir="/data/REID/faces",
                           video_dir="/data/REID/frames",
                           image_size=112,
                           is_train=False,
                           train_split=587)
    dataloader = DataLoader(dataset,
                            batch_size=8,
                            num_workers=4,
                            pin_memory=True)
    model.cuda()
    model.eval()

    embeds_list, label_list = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            data = [batch[0][0].to("cuda"), batch[0][1].to("cuda")]
            label_list.append(batch[1])
            embeds_list.append(F.normalize(model(data)).to("cpu"))
        torch.save([torch.concat(embeds_list), torch.concat(label_list)],
                   ckpt.split(".ck")[0] + ".embds")
        # print("{} embeds saved".format(torch.concat(embeds_list).shape))


def get_mm2_embds2(cfg: str, ckpt: str):
    model = MMClassification.load_from_checkpoint(ckpt,
                                                  args=OmegaConf.load(cfg).experiments.model)
    dataset = UnionDataset3(face_dir="/data/REID/faces",
                            video_dir="/data/REID/frames",
                            audio_dir="/data/REID/audios",
                            image_size=112,
                            is_train=False,
                            train_split=587)
    dataloader = DataLoader(dataset,
                            batch_size=32,
                            num_workers=8,
                            pin_memory=True)
    model.cuda()
    model.eval()

    embeds_list, label_list = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            data = [batch[0][0].to("cuda"), batch[0][1].to("cuda")]
            label_list.append(batch[1])
            embeds_list.append(F.normalize(model(data)).to("cpu"))
        torch.save([torch.concat(embeds_list), torch.concat(label_list)],
                   ckpt.split(".ck")[0] + ".embds2")
        # print("{} embeds saved".format(torch.concat(embeds_list).shape))


def get_mm3_embds(cfg: str, ckpt: str):
    model = MFFN_DBFF.load_from_checkpoint(ckpt,
                                           args=OmegaConf.load(cfg).experiments.model,
                                           strict=False)

    dataset = UnionDataset3(face_dir="/data/REID/faces",
                            video_dir="/data/REID/frames",
                            audio_dir="/data/REID/audios",
                            image_size=112,
                            is_train=False,
                            train_split=587)
    dataloader = DataLoader(dataset,
                            batch_size=8,
                            num_workers=4,
                            pin_memory=False)
    model.cuda()
    model.eval()

    embeds_list, label_list = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            data = [batch[0][0].to("cuda"), batch[0][1].to("cuda"), batch[0][2].to("cuda")]
            label_list.append(batch[1])
            embeds_list.append(F.normalize(model(data)).to("cpu"))
        torch.save([torch.concat(embeds_list), torch.concat(label_list)],
                   ckpt.split(".ck")[0] + ".embds")


def get_mm3_attVLAD_embds(cfg: str, ckpt: str):
    model = MFFN_attVALD1.load_from_checkpoint(ckpt,
                                               args=OmegaConf.load(cfg).experiments.model,
                                               strict=False)
    dataset = UnionDataset3_MLMA(face_dir="/data/REID/faces",
                                 video_dir="/data/REID/frames",
                                 audio_dir="/data/REID/audios",
                                 image_size=112,
                                 is_train=False,
                                 train_split=587)
    dataloader = DataLoader(dataset,
                            batch_size=8,
                            num_workers=4,
                            pin_memory=False)
    model.cuda()
    model.eval()

    embeds_list, label_list = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            data = [batch[0][0].to("cuda"), batch[0][1].to("cuda"), batch[0][2].to("cuda")]
            label_list.append(batch[1])
            embeds_list.append(F.normalize(model(data)).to("cpu"))
        torch.save([torch.concat(embeds_list), torch.concat(label_list)],
                   ckpt.split(".ck")[0] + ".embds")


def get_mm3_MLMDA_embds(cfg: str, ckpt: str):
    model = ML_MDA.load_from_checkpoint(ckpt,
                                        args=OmegaConf.load(cfg).experiments.model,
                                        strict=False)
    dataset = UnionDataset3_MLMDA(face_dir="/data/REID/faces",
                                  video_dir="/data/REID/frames",
                                  audio_dir="/data/REID/audios",
                                  image_size=112,
                                  is_train=False,
                                  train_split=587)
    dataloader = DataLoader(dataset,
                            batch_size=8,
                            num_workers=4,
                            pin_memory=False)
    model.cuda()
    model.eval()

    embeds_list, label_list = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            data = [batch[0][0].to("cuda"), batch[0][1].to("cuda"), batch[0][2].to("cuda")]
            label_list.append(batch[1])
            embeds_list.append(F.normalize(model(data)).to("cpu"))
        torch.save([torch.concat(embeds_list), torch.concat(label_list)],
                   ckpt.split(".ck")[0] + ".embds")


def get_audio_embds(cfg: str, ckpt: str):
    model = RawNet_NL_GRU()
    model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)
    dataset = UnionDataset3(face_dir="/data/REID/faces",
                            video_dir="/data/REID/frames",
                            audio_dir="/data/REID/audios",
                            image_size=112,
                            is_train=False,
                            train_split=587)
    dataloader = DataLoader(dataset,
                            batch_size=32,
                            num_workers=8,
                            pin_memory=False)
    model.cuda()
    model.eval()

    embeds_list, label_list = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            data = batch[0][2].to("cuda")
            label_list.append(batch[1])
            embeds_list.append(F.normalize(model(data)).to("cpu"))
        torch.save([torch.concat(embeds_list), torch.concat(label_list)],
                   "/home/yjc/PythonProject/PLIntegration/train_outputs/mma_reid/rawnet.embds")


def get_visual_results(cfg: str, ckpt: str):
    model = MMClassification_XY_YZ1.load_from_checkpoint(ckpt,
                                                         args=OmegaConf.load(cfg).experiments.model,
                                                         strict=False)
    dataset = UnionDataset3(face_dir="/data/REID/faces",
                            video_dir="/data/REID/frames",
                            audio_dir="/data/REID/audios",
                            image_size=112,
                            is_train=False,
                            train_split=587,
                            get_visual_results=True)
    dataloader = DataLoader(dataset,
                            batch_size=8,
                            num_workers=4,
                            pin_memory=False)
    model.cuda()
    model.eval()

    embeds_list, label_list, path_list = [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            data = [batch[0][0].to("cuda"), batch[0][1].to("cuda"), batch[0][2].to("cuda")]
            label_list.append(batch[1])
            path_list.extend(batch[2])
            embeds_list.append(F.normalize(model(data)).to("cpu"))
        torch.save([torch.concat(embeds_list), torch.concat(label_list), path_list],
                   ckpt.split(".ck")[0] + ".embds_results")


def save_visual_resutls(embeds_path: str):
    embeds, labels, img_path = torch.load(embeds_path)
    embeds = embeds.cuda()
    sim_matrix = []
    embeds_split = torch.chunk(embeds, 50)

    with torch.no_grad():
        for row in tqdm(range(50)):
            sim = torch.matmul(embeds_split[row], embeds.T)
            sim_matrix.append(torch.argsort(sim, 1, True)[:, 1:21].cpu())
    torch.save([torch.concat(sim_matrix, 0), labels, img_path], embeds_path.split(".em")[0] + ".visual_results")


def get_sim_matrix(embeds_path: str):
    embeds, labels = torch.load(embeds_path)
    embeds = embeds.cuda()
    sim_matrix = []
    embeds_split = torch.chunk(embeds, 50)

    with torch.no_grad():
        for row in tqdm(range(50)):
            sim = torch.matmul(embeds_split[row], embeds.T)
            sim_matrix.append(torch.argsort(sim, 1, True)[:, 1:21].cpu())
    torch.save([torch.concat(sim_matrix, 0), labels], embeds_path.split(".em")[0] + ".sim")


def CMC_all(embeds_path: str, ranks: int = 20):
    embeds, labels = torch.load(embeds_path)
    embeds = embeds.cuda()
    sim_matrix = []
    embeds_split = torch.chunk(embeds, 50)

    with torch.no_grad():
        for row in tqdm(range(50)):
            sim = torch.matmul(embeds_split[row], embeds.T)
            sim_matrix.append(torch.argsort(sim, 1, True)[:, 1:ranks + 1].cpu())
    del embeds, embeds_split
    sim_matrix = torch.concat(sim_matrix, 0)
    torch.save([sim_matrix, labels], embeds_path.split(".em")[0] + ".sim")

    match_matrix = torch.zeros_like(sim_matrix)
    for row in tqdm(range(sim_matrix.shape[0])):
        for col in range(ranks):
            if labels[row] == labels[sim_matrix[row, col]]:
                match_matrix[row, col:] += 1
                break
    CMC = torch.mean(match_matrix.float(), 0)
    print("CMC@R1={}\nCMC@R3={}\nCMC@R5={}\nCMC@R10={}\n".format(CMC[0], CMC[2], CMC[4], CMC[9]))
    print("ALL CMC = {}".format(CMC))
    print(embeds_path)
    del sim_matrix, match_matrix, labels, CMC


def mAP_all(embeds_path: str):
    embeds, labels = torch.load(embeds_path)
    embeds = embeds.cuda()
    sim_matrix = []
    embeds_split = torch.chunk(embeds, 50)
    # labels = torch.split(labels, 5)
    with torch.no_grad():
        for row in tqdm(range(50)):
            sim = torch.matmul(embeds_split[row], embeds.T)
            sim_matrix.append(torch.argsort(sim, 1, True)[:, :].cpu())
        del embeds, embeds_split
        sim_matrix = torch.cat(sim_matrix, 0)

        match_matrix = torch.zeros(sim_matrix.shape[1]).cuda()
        APs = []
        for row in tqdm(range(sim_matrix.shape[0])):
            torch.zero_(match_matrix)
            match_matrix[torch.where(labels[sim_matrix[row]] == labels[row])] = 1
            non_zero = torch.nonzero(match_matrix).squeeze().cpu() + 1
            try:
                APs.append(torch.mean(torch.arange(1, non_zero.shape[0] + 1) / non_zero))
            except IndexError:
                APs.append(torch.mean(1 / non_zero))
                # print(non_zero)

        # for col in range(sim_matrix.shape[1]):
        #     if labels[row] == labels[sim_matrix[row, col]]:
        #         match_matrix[row, col:] += 1
        #     else:
        #         match_matrix[row, col] = 0

        # APs = torch.sum(match_matrix / torch.arange(1, sim_matrix.shape[1] + 1), 1) / torch.count_nonzero(match_matrix, 1)
        mAP = torch.mean(torch.tensor(APs))
        print("mAP = {}".format(mAP))
        print(embeds_path)


def CMC_single(embeds_path: str):
    sim_matrix, labels = torch.load(sim_path)
    match_matrix = torch.zeros_like(sim_matrix)
    for row in tqdm(range(sim_matrix.shape[0])):
        for col in range(20):
            if labels[row] == labels[sim_matrix[row, col]]:
                match_matrix[row, col:] += 1
                break
    CMC = torch.mean(match_matrix.float(), 0)
    print("CMC@R1={}\nCMC@R3={}\nCMC@R5={}\nCMC@R10={}\n".format(CMC[0], CMC[2], CMC[4], CMC[9]))
    print("ALL CMC = {}".format(CMC))


def print_CMCs(embeds_pathes: list, ranks: int = 100):
    fig = plt.figure(figsize=(25.60, 19.20))
    plt.rc('font', family='Times New Roman', size=36)
    plt.xlabel("Rank")
    plt.ylabel("Matching rate")
    plt.ylim(0.99, 1)
    # plt.legend(loc="best")
    # plt.title("CMC Curve")

    CMCs = []
    for embeds_path in embeds_pathes:
        embeds, labels = torch.load(embeds_path)
        # embeds = embeds.cuda()
        sim_matrix = []
        embeds_split = torch.chunk(embeds, 50)

        with torch.no_grad():
            for row in tqdm(range(50)):
                sim = torch.matmul(embeds_split[row], embeds.T)
                sim_matrix.append(torch.argsort(sim, 1, True)[:, 1:ranks + 1].cpu())
        del embeds, embeds_split
        sim_matrix = torch.concat(sim_matrix, 0)
        torch.save([sim_matrix, labels], embeds_path.split(".em")[0] + ".sim")

        match_matrix = torch.zeros_like(sim_matrix)
        for row in tqdm(range(sim_matrix.shape[0])):
            for col in range(ranks):
                if labels[row] == labels[sim_matrix[row, col]]:
                    match_matrix[row, col:] += 1
                    break
        CMC = torch.mean(match_matrix.float(), 0)
        CMCs.append(CMC)

    color = ["b", "y", "g", "c", "purple", "r", "darkorange", "slategray", "deeppink"]
    lines = ["-", "--", "-.", ":", "-.", "-", "--", ":", "-."]
    labels = ["CBP", "LBP", "FA-MFF", "ML-MDA", "i-Code", "LCMA-Net*", "AB-LCMPA", "AF+FB-LCMPA*",
              "AF+FB-LCMPA"]

    for i in range(len(CMCs)):
        x = np.arange(1, len(CMCs[i]))
        locals()["line{}".format(i)], = plt.plot(x, CMCs[i][:-1], linestyle=lines[i], color=color[i], linewidth=2)
    plt.legend(handles=[locals()["line0"],
                        locals()["line1"],
                        locals()["line2"],
                        locals()["line3"],
                        locals()["line4"],
                        locals()["line5"],
                        locals()["line6"],
                        locals()["line7"],
                        locals()["line8"]], labels=labels)
    plt.savefig("/mnt/homeold/yjc/PythonProject/PLIntegration/cmc_curves_new.png")
    plt.show()


def get_gflops(cfg: str, ckpt: str):
    model = MMClassification_YZ1.load_from_checkpoint(ckpt,
                                                      args=OmegaConf.load(cfg).experiments.model,
                                                      strict=False)
    # model = MMClassification.load_from_checkpoint(ckpt,
    #                                               args=OmegaConf.load(cfg).experiments.model,
    #                                               strict=False)
    model.to("cuda:0")

    input_t = [[
        torch.rand((2, 3, 112, 112)).to("cuda:0"),
        torch.rand((2, 3, 16, 112, 112)).to("cuda:0"),
        torch.rand((2, 59049)).to("cuda:0"),
    ]]

    flops, prams = profile(model, input_t)
    flops, prams = clever_format([flops, prams], '%.3f')
    print(flops, prams)


if __name__ == '__main__':
    PATH_PREFIX = "/mnt/homeold/yjc/PythonProject/PLIntegration/"
    # region    # <--- MM2 ---> #
    get_mm3_embds(
        "train_outputs/.../.hydra/config.yaml",
        "train_outputs/.../checkpoints/epoch=189-train_loss=0.14-EER=2.05.ckpt")
    # endregion # <--- MM3 ---> #

    # region    # <--- CMC&mAP ---> #

    CMC_all(
        "train_outputs/.../checkpoints/epoch=189-train_loss=0.14-EER=2.05.embds")
    mAP_all(
        "train_outputs/.../checkpoints/epoch=189-train_loss=0.14-EER=2.05.embds")
    # endregion # <--- CMC&mAP ---> #

    # get_gflops(
    #         .../.hydra/config.yaml",
    #         .../checkpoints/epoch=159-train_loss=0.11-EER=2.94.ckpt"
    # )
    pass

