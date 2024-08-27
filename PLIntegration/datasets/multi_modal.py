#!usr/bin/env python
# -*- coding:utf-8 _*-
import os
import os.path as osp
from typing import Union

import numpy as np
import torch
import torchaudio
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from PLIntegration.datasets.audio import get_audio_list, get_audio_dic, processed, tta_processed
from PLIntegration.datasets.video import get_list_label_reid


def get_img_list(src_dir):
    l_img = []
    for path, dirs, files in tqdm(os.walk(src_dir), desc="get_img_list " + src_dir):
        base = path.split("/")[-1] + "/"
        for file in files:
            if file[-3:] != 'jpg' and file[-3:] != 'png':
                continue
            l_img.append(base + file)
    return l_img


def get_img_dic(l_img):
    face_dic = {}
    for img in tqdm(l_img, desc="get_dic"):
        spk = img.split("/")[0]
        if spk not in face_dic:
            face_dic[spk] = [img]
        else:
            face_dic[spk].append(img)
    return face_dic


class UnionDataset(Dataset):
    def __init__(self, face_dir, video_dir, image_size, train_split=587, is_train=True,
                 **kwargs):
        '''
        self.list_IDs	: list of strings (each string: utt key)
        self.labels		: dictionary (key: utt key, value: label integer)
        train_split	    :     20%:122    40%:259    80%:587
        '''
        l_dev_face = sorted(get_img_list(face_dir))
        d_dev_face = get_img_dic(l_dev_face)
        l_dev_motion, d_label_union = get_list_label_reid(video_dir)
        l_dev_union, l_val_union = [], []

        for motion in l_dev_motion:
            key = motion[0].split("/")[0]
            faces = d_dev_face[key]
            if int(key) >= train_split:
                l_val_union.append([motion, faces])
            else:
                l_dev_union.append([motion, faces])
        if is_train:
            self.list_IDs = l_dev_union
        else:
            self.list_IDs = l_val_union
        self.video_dir = video_dir
        self.face_dir = face_dir
        self.labels = d_label_union
        if is_train:
            self.transformer = {
                "face":  transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]),
                "video": transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor()
                ]),
            }
        else:
            self.transformer = {
                "face":  transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ]),
                "video": transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor()
                ]),
            }

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        IDs = self.list_IDs[index]
        try:
            frames = []
            for ID in IDs[0]:
                x_video = Image.open(osp.join(self.video_dir, ID))
                x_video = self.transformer["video"](x_video)
                frames.append(x_video)
            x_video = torch.stack(frames, 1) / 255
        except Exception:
            raise ValueError('%s' % ID)

        try:
            ID = IDs[1][np.random.randint(0, len(IDs[1]))]
            x_face = Image.open(osp.join(self.face_dir, ID))
            x_face = self.transformer["face"](x_face)
        except Exception:
            raise ValueError('%s' % ID)

        y = self.labels[IDs[0][0].split("/")[0]]

        return [x_face, x_video], y


class TrialTestOld(UnionDataset):
    def __init__(self, face_dir, video_dir, image_size, train_split=587, is_train=False, **kwargs):
        super().__init__(face_dir, video_dir, image_size, train_split, is_train, **kwargs)

    def __getitem__(self, index):
        IDs = self.list_IDs[index]
        try:
            frames = []
            for ID in IDs[0]:
                x_video = Image.open(osp.join(self.video_dir, ID))
                x_video = self.transformer["video"](x_video)
                frames.append(x_video)
            x_video = torch.stack(frames, 1) / 255
        except Exception:
            raise ValueError('%s' % ID)

        try:
            ID = IDs[1][np.random.randint(0, len(IDs[1]))]
            x_face = Image.open(osp.join(self.face_dir, ID))
            x_face = self.transformer["face"](x_face)
        except Exception:
            raise ValueError('%s' % ID)

        y = self.labels[IDs[0][0].split("/")[0]]
        return [x_face, x_video], y, self.list_IDs[index][0][0]


class UnionDataset3(Dataset):
    def __init__(self, face_dir, video_dir, audio_dir, image_size, nb_samp=59049, train_split=587, is_train=True
                 , get_visual_results=False, **kwargs):
        '''
        self.list_IDs	: list of strings (each string: utt key)
        self.labels		: dictionary (key: utt key, value: label integer)
        train_split	    :     20%:122    40%:259    80%:587
        '''
        l_dev_face = sorted(get_img_list(face_dir))
        d_dev_face = get_img_dic(l_dev_face)
        l_dev_motion, d_label_union = get_list_label_reid(video_dir)
        l_dev_union, l_val_union = [], []
        l_dev_audio = sorted(get_audio_list(audio_dir))
        d_dev_audio = get_audio_dic(l_dev_audio)
        for motion in l_dev_motion:
            key = motion[0].split("/")[0]
            faces = d_dev_face[key]
            audio = d_dev_audio[key]
            audio_dic = {}
            for a in audio:
                audio_dic[osp.basename(a).split(".")[0]] = a
            if int(key) >= train_split:
                l_val_union.append([motion, faces, audio_dic])
            else:
                l_dev_union.append([motion, faces, audio_dic])
        if is_train:
            self.list_IDs = l_dev_union
        else:
            self.list_IDs = l_val_union
        self.video_dir = video_dir
        self.face_dir = face_dir
        self.audio_dir = audio_dir
        self.labels = d_label_union
        self.nb_samp = nb_samp
        self.is_train = is_train
        self.get_visual_results = get_visual_results
        if is_train:
            self.transformer = {
                "face":  transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]),
                "video": transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor()
                ]),
            }
        else:
            self.transformer = {
                "face":  transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ]),
                "video": transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor()
                ]),
            }

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        IDs = self.list_IDs[index]
        try:
            frames = []
            for ID in IDs[0]:
                x_video = Image.open(osp.join(self.video_dir, ID))
                x_video = self.transformer["video"](x_video)
                frames.append(x_video)
            x_video = torch.stack(frames, 1) / 255
        except Exception:
            raise ValueError('%s' % ID)

        try:
            ID = IDs[2][osp.basename(IDs[0][0])[:-10]]
            x_audio, _ = processed(ID, self.audio_dir, self.nb_samp)
        except Exception:
            raise ValueError('%s' % ID)

        try:
            ID = IDs[1][np.random.randint(0, len(IDs[1]))]
            x_face = Image.open(osp.join(self.face_dir, ID))
            x_face = self.transformer["face"](x_face)
        except Exception:
            raise ValueError('%s' % ID)

        y = self.labels[IDs[0][0].split("/")[0]]

        if self.get_visual_results:
            return [x_face, x_video, x_audio], y, osp.join(self.video_dir, IDs[0][0])

        return [x_face, x_video, x_audio], y


class UnionDataset3_MLMDA(Dataset):
    def __init__(self, face_dir, video_dir, audio_dir, image_size, nb_samp=59049, train_split=587, is_train=True
                 , get_visual_results=False, **kwargs):
        '''
        self.list_IDs	: list of strings (each string: utt key)
        self.labels		: dictionary (key: utt key, value: label integer)
        train_split	    :     20%:122    40%:259    80%:587
        '''
        l_dev_face = sorted(get_img_list(face_dir))
        d_dev_face = get_img_dic(l_dev_face)
        l_dev_motion, d_label_union = get_list_label_reid(video_dir)
        l_dev_union, l_val_union = [], []
        l_dev_audio = sorted(get_audio_list(audio_dir))
        d_dev_audio = get_audio_dic(l_dev_audio)
        for motion in l_dev_motion:
            key = motion[0].split("/")[0]
            faces = d_dev_face[key]
            audio = d_dev_audio[key]
            audio_dic = {}
            for a in audio:
                audio_dic[osp.basename(a).split(".")[0]] = a
            if int(key) >= train_split:
                l_val_union.append([motion, faces, audio_dic])
            else:
                l_dev_union.append([motion, faces, audio_dic])
        if is_train:
            self.list_IDs = l_dev_union
        else:
            self.list_IDs = l_val_union
        self.video_dir = video_dir
        self.face_dir = face_dir
        self.audio_dir = audio_dir
        self.labels = d_label_union
        self.nb_samp = nb_samp
        self.is_train = is_train
        self.get_visual_results = get_visual_results
        if is_train:
            self.transformer = {
                "face":  transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]),
                "video": transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor()
                ]),
                "audio": torchaudio.transforms.Spectrogram(
                    n_fft=512,
                    win_length=None,
                    hop_length=256,
                    center=True,
                    pad_mode="reflect",
                    power=2.0,
                )
            }
        else:
            self.transformer = {
                "face":  transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ]),
                "video": transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor()
                ]),
                "audio": torchaudio.transforms.Spectrogram(
                    n_fft=512,
                    win_length=None,
                    hop_length=256,
                    center=True,
                    pad_mode="reflect",
                    power=2.0,
                )
            }

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        IDs = self.list_IDs[index]
        try:
            frames = []
            for ID in IDs[0]:
                x_video = Image.open(osp.join(self.video_dir, ID))
                x_video = self.transformer["video"](x_video)
                # black_box = torch.zeros_like(x_video)
                # randx = np.random.randint(112 - 10)
                # randy = np.random.randint(112 - 10)
                # x_video[:, randx:randx + 10, randy:randy + 10] = black_box[:, randx:randx + 10, randy:randy + 10]
                # if np.random.rand() < 0.05:
                #     x_video = black_box
                break
        except Exception:
            raise ValueError('%s' % ID)

        try:
            ID = IDs[2][osp.basename(IDs[0][0])[:-10]]
            x_audio, _ = processed(ID, self.audio_dir, self.nb_samp)
            x_audio = torch.repeat_interleave(self.transformer["audio"](x_audio).unsqueeze(0), 3, 0)
            # try:
            #     black_box = torch.zeros_like(x_audio)
            #     randx = np.random.randint(x_audio.shape[1] - 0.1 * x_audio.shape[1])
            #     x_audio[:, randx:randx + int(0.1 * x_audio.shape[1]), :] = black_box[:, randx:randx + int(0.1 * x_audio.shape[1]), :]
            #     if np.random.rand() < 0.05:
            #         x_audio = black_box
            # except Exception:
            #     raise ValueError('%s' % ID)

        except Exception:
            raise ValueError('%s' % ID)

        try:
            ID = IDs[1][np.random.randint(0, len(IDs[1]))]
            x_face = Image.open(osp.join(self.face_dir, ID))
            x_face = self.transformer["face"](x_face)
            # black_box = torch.zeros_like(x_face)
            # randx = np.random.randint(112 - 10)
            # randy = np.random.randint(112 - 10)
            # x_face[:, randx:randx + 10, randy:randy + 10] = black_box[:, randx:randx + 10, randy:randy + 10]
            # if np.random.rand() < 0.05:
            #     x_face = black_box
        except Exception:
            raise ValueError('%s' % ID)

        y = self.labels[IDs[0][0].split("/")[0]]

        if self.get_visual_results:
            return [x_face, x_video, x_audio], y, osp.join(self.video_dir, IDs[0][0])

        return [x_face, x_video, x_audio], y


class TrialTestOld3(UnionDataset3):
    def __init__(self, face_dir, video_dir, audio_dir, image_size, nb_samp=59049, train_split=587, is_train=True,
                 **kwargs):
        super().__init__(face_dir, video_dir, audio_dir, image_size, nb_samp, train_split, is_train, **kwargs)

    def __getitem__(self, index):
        IDs = self.list_IDs[index]
        try:
            frames = []
            for ID in IDs[0]:
                x_video = Image.open(osp.join(self.video_dir, ID))
                x_video = self.transformer["video"](x_video)
                frames.append(x_video)
            x_video = torch.stack(frames, 1) / 255
        except Exception:
            raise ValueError('%s' % ID)

        try:
            ID = IDs[2][osp.basename(IDs[0][0])[:-10]]
            x_audio, _ = processed(ID, self.audio_dir, self.nb_samp)
        except Exception:
            raise ValueError('%s' % ID)

        try:
            ID = IDs[1][np.random.randint(0, len(IDs[1]))]
            x_face = Image.open(osp.join(self.face_dir, ID))
            x_face = self.transformer["face"](x_face)
        except Exception:
            raise ValueError('%s' % ID)
        y = self.labels[IDs[0][0].split("/")[0]]
        return [x_face, x_video, x_audio], y, self.list_IDs[index][0][0]


class UnionDataset3_MLMA(Dataset):
    def __init__(self, face_dir, video_dir, audio_dir, image_size, nb_samp=59049, train_split=587, is_train=True
                 , get_visual_results=False, **kwargs):
        '''
        self.list_IDs	: list of strings (each string: utt key)
        self.labels		: dictionary (key: utt key, value: label integer)
        train_split	    :     20%:122    40%:259    80%:587
        '''
        l_dev_face = sorted(get_img_list(face_dir))
        d_dev_face = get_img_dic(l_dev_face)
        l_dev_motion, d_label_union = get_list_label_reid(video_dir)
        l_dev_union, l_val_union = [], []
        l_dev_audio = sorted(get_audio_list(audio_dir))
        d_dev_audio = get_audio_dic(l_dev_audio)
        for motion in l_dev_motion:
            key = motion[0].split("/")[0]
            faces = d_dev_face[key]
            audio = d_dev_audio[key]
            audio_dic = {}
            for a in audio:
                audio_dic[osp.basename(a).split(".")[0]] = a
            if int(key) >= train_split:
                l_val_union.append([motion, faces, audio_dic])
            else:
                l_dev_union.append([motion, faces, audio_dic])
        if is_train:
            self.list_IDs = l_dev_union
        else:
            self.list_IDs = l_val_union
        self.video_dir = video_dir
        self.face_dir = face_dir
        self.audio_dir = audio_dir
        self.labels = d_label_union
        self.nb_samp = nb_samp
        self.is_train = is_train
        self.get_visual_results = get_visual_results
        if is_train:
            self.transformer = {
                "face":  transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]),
                "video": transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor()
                ]),
            }
        else:
            self.transformer = {
                "face":  transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ]),
                "video": transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor()
                ]),
            }

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        IDs = self.list_IDs[index]
        try:
            frames = []
            for ID in IDs[0]:
                x_video = Image.open(osp.join(self.video_dir, ID))
                x_video = self.transformer["video"](x_video)
                frames.append(x_video)
            x_video = torch.stack(frames, 1) / 255
        except Exception:
            raise ValueError('%s' % ID)

        try:
            ID = IDs[2][osp.basename(IDs[0][0])[:-10]]
            x_audio, _ = processed(ID, self.audio_dir, self.nb_samp)
        except Exception:
            raise ValueError('%s' % ID)

        try:
            faces = []
            for i in np.random.randint(0, len(IDs[1]), 16):
                ID = IDs[1][i]
                x_face = Image.open(osp.join(self.face_dir, ID))
                faces.append(self.transformer["face"](x_face))
            x_face = torch.stack(faces, 0)
        except Exception:
            raise ValueError('%s' % ID)

        y = self.labels[IDs[0][0].split("/")[0]]

        if self.get_visual_results:
            return [x_face, x_video, x_audio], y, osp.join(self.video_dir, IDs[0][0])

        return [x_face, x_video, x_audio], y


class TrialTestOld3_MLMA(UnionDataset3_MLMA):
    def __init__(self, face_dir, video_dir, audio_dir, image_size, nb_samp=59049, train_split=587, is_train=True,
                 **kwargs):
        super().__init__(face_dir, video_dir, audio_dir, image_size, nb_samp, train_split, is_train, **kwargs)

    def __getitem__(self, index):
        IDs = self.list_IDs[index]
        try:
            frames = []
            for ID in IDs[0]:
                x_video = Image.open(osp.join(self.video_dir, ID))
                x_video = self.transformer["video"](x_video)
                frames.append(x_video)
            x_video = torch.stack(frames, 1) / 255
        except Exception:
            raise ValueError('%s' % ID)

        try:
            ID = IDs[2][osp.basename(IDs[0][0])[:-10]]
            x_audio, _ = processed(ID, self.audio_dir, self.nb_samp)
        except Exception:
            raise ValueError('%s' % ID)

        try:
            faces = []
            for i in np.random.randint(0, len(IDs[1]), 16):
                ID = IDs[1][i]
                x_face = Image.open(osp.join(self.face_dir, ID))
                faces.append(self.transformer["face"](x_face))
            x_face = torch.stack(faces, 0)
        except Exception:
            raise ValueError('%s' % ID)
        y = self.labels[IDs[0][0].split("/")[0]]
        return [x_face, x_video, x_audio], y, self.list_IDs[index][0][0]


class TrialTestOld3_MLMDA(UnionDataset3_MLMDA):
    def __init__(self, face_dir, video_dir, audio_dir, image_size, nb_samp=59049, train_split=587, is_train=True,
                 **kwargs):
        super().__init__(face_dir, video_dir, audio_dir, image_size, nb_samp, train_split, is_train, **kwargs)

    def __getitem__(self, index):
        IDs = self.list_IDs[index]
        try:
            frames = []
            for ID in IDs[0]:
                x_video = Image.open(osp.join(self.video_dir, ID))
                x_video = self.transformer["video"](x_video)
                break
        except Exception:
            raise ValueError('%s' % ID)

        try:
            ID = IDs[2][osp.basename(IDs[0][0])[:-10]]
            x_audio, _ = processed(ID, self.audio_dir, self.nb_samp)
            x_audio = torch.repeat_interleave(self.transformer["audio"](x_audio).unsqueeze(0), 3, 0)
        except Exception:
            raise ValueError('%s' % ID)

        try:
            ID = IDs[1][np.random.randint(0, len(IDs[1]))]
            x_face = Image.open(osp.join(self.face_dir, ID))
            x_face = self.transformer["face"](x_face)
        except Exception:
            raise ValueError('%s' % ID)
        y = self.labels[IDs[0][0].split("/")[0]]
        return [x_face, x_video, x_audio], y, self.list_IDs[index][0][0]


if __name__ == '__main__':
    dataset = UnionDataset3_MLMDA(face_dir="/data/REID/faces",
                                  video_dir="/data/REID/frames",
                                  audio_dir="/data/REID/audios",
                                  image_size=112,
                                  is_train=True,
                                  train_split=587)
    # dataloader = DataLoader(dataset,
    #                         batch_size=8,
    #                         num_workers=0,
    #                         pin_memory=True)

    for batch in tqdm(dataset):
        print(len(batch))