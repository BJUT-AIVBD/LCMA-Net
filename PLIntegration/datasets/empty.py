#!usr/bin/env python
# -*- coding:utf-8 _*-
from torch.utils.data import Dataset


class Empty(Dataset):
    def __init__(self, **kwargs):
        super(Empty, self).__init__()

    def __len__(self):
        return 0

    def __getitem__(self, item):
        return None
