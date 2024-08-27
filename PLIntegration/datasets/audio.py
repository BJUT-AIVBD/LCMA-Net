#!usr/bin/env python
# -*- coding:utf-8 _*-
import os
import os.path as osp

import torch
import torchaudio
from tqdm import tqdm
import numpy as np


def get_audio_list(src_dir, cnceleb=False):
    '''
    Designed for VoxCeleb
    '''
    l_utt = []
    for path, dirs, files in tqdm(os.walk(src_dir), desc="get_utt " + src_dir):
        base = path.split("/")[-1] + "/"
        for file in files:
            if file[-3:] != 'mp3' and file[-3:] != 'wav' and file[-4:] != 'flac' and file[-3:] != 'npy':
                continue
            l_utt.append(base + file)
    return l_utt


def get_audio_dic(l_utt):
    d_label = {}
    for utt in tqdm(l_utt, desc="get_dic"):
        spk = utt.split('/')[0]
        if spk not in d_label:
            d_label[spk] = [utt]
        else:
            d_label[spk].append(utt)
    return d_label


def processed(ID, base_dir, nb_samp, labels=None):
    x_audio, _ = torchaudio.backend.sox_io_backend.load(osp.join(base_dir, ID), normalize=False)
    x_audio = x_audio.numpy()
    x_audio = x_audio / np.max(np.abs(x_audio))
    x_audio = x_audio.reshape(1, -1)  # because of LayerNorm for the input

    nb_time = x_audio.shape[1]
    if nb_time > nb_samp:
        start_idx = np.random.randint(low=0, high=nb_time - nb_samp)
        x_audio = x_audio[:, start_idx: start_idx + nb_samp][0]
    elif nb_time < nb_samp:
        nb_dup = int(nb_samp / nb_time) + 1
        x_audio = np.tile(x_audio, (1, nb_dup))[:, :nb_samp][0]
    else:
        x_audio = x_audio[0]
    if labels is None:
        y = None
    else:
        y = labels[ID.split('/')[0]]
    return torch.FloatTensor(x_audio), y


def tta_processed(ID, base_dir, nb_samp, labels=None, window_size=0):
    x_audio, _ = torchaudio.backend.sox_io_backend.load(osp.join(base_dir, ID), normalize=False)
    x_audio = x_audio.numpy()
    x_audio = x_audio / np.max(np.abs(x_audio))
    x_audio = x_audio.reshape(1, -1)  # because of LayerNorm for the input

    list_X = []
    nb_time = x_audio.shape[1]
    if nb_time < nb_samp:
        nb_dup = int(nb_samp / nb_time) + 1
        list_X.append(np.tile(x_audio, (1, nb_dup))[:, :nb_samp][0])
    elif nb_time > nb_samp:
        step = nb_samp - window_size
        iteration = int((nb_time - window_size) / step) + 1
        for i in range(iteration):
            if i == 0:
                list_X.append(x_audio[:, :nb_samp][0])
            elif i < iteration - 1:
                list_X.append(x_audio[:, i * step: i * step + nb_samp][0])
            else:
                list_X.append(x_audio[:, -nb_samp:][0])
    else:
        list_X.append(x_audio[0])

    if labels is None:
        y = None
    else:
        y = labels[ID.split('/')[0]]
    return list_X, y