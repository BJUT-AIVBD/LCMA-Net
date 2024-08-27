#!usr/bin/env python
# -*- coding:utf-8 _*-
import numbers
import os
import os.path as osp
from typing import Union

import mxnet as mx
import numpy as np
import torch
import torchvision.transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from PLIntegration.datasets.utils.manifest import loader


class ManifestBase(Dataset):
    def __init__(self, manifest_filepath: str,
                 transformer: Union[torchvision.transforms.Compose, None] = None,
                 **kwargs):
        super().__init__()
        self.img_files, self.labels, _ = loader.load(manifest_filepath, "face")
        self.transformer = transformer

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = Image.open(self.img_files[idx])
        label = self.labels[idx]
        if self.transformer is not None:
            img = self.transformer(img)
        else:
            img = torchvision.transforms.ToTensor()(img)

        return img, label


class ManifestSave(ManifestBase):
    def __init__(self, manifest_path, transformers, output_dir, overwrite):
        super(ManifestSave, self).__init__(manifest_path, transformers)
        self.output_dir = output_dir
        self.overwrite = overwrite

    def __getitem__(self, idx):
        output_path = osp.join(self.output_dir, "/".join(self.img_files[idx].split("/")[-2:])[:-3] + "jpg")
        if not self.overwrite:
            if osp.exists(output_path):
                return torch.tensor(0)
        img, _ = super().__getitem__(idx)
        if not osp.exists(osp.dirname(output_path)):
            try:
                os.makedirs(osp.dirname(output_path))
            except Exception:
                print("create {} failed".format(osp.dirname(output_path)))
        try:
            img.save(output_path, quality=95)
        except Exception:
            print("Save {} failed!".format(output_path))
            if osp.exists(output_path):
                os.remove(output_path)
                print("Delete broken file {}!".format(output_path))
        return torch.tensor(0)


class TrialTest(ManifestBase):
    def __init__(self, transformer: Union[torchvision.transforms.Compose, None] = None,
                 **kwargs):
        super().__init__(transformer=transformer, **kwargs)

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        return img, label, self.img_files[idx]


class MxRecBase(Dataset):
    def __init__(self, manifest_filepath,
                 transformer: Union[torchvision.transforms.Compose, None] = None,
                 **kwargs):
        super().__init__()
        # self.transform = transforms.Compose(
        #     [transforms.ToPILImage(),
        #      transforms.RandomHorizontalFlip(),
        #      transforms.ToTensor(),
        #      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        #      ])
        self.transform = transformer
        self.root_dir = manifest_filepath
        path_imgrec = os.path.join(manifest_filepath, 'train.rec')
        path_imgidx = os.path.join(manifest_filepath, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)
