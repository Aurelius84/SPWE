#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/22 下午7:06
# @From    : PyCharm
# @File    : transforms
# @Author  : Liujie Zhang
# @Email   : liujiezhangbupt@gmail.com

import torch


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class Pad(object):

    def __init__(self, seq_len, default=0):
        self.seq_len = seq_len
        self.default = default

    def __call__(self, data):
        if len(data) >= self.seq_len:
            return data[:self.seq_len]
        else:
            return data + [self.default] * (self.seq_len - len(data))


class ToIndex(object):

    def __init__(self, lookup, default=0):
        self.lookup = lookup
        self.default = default

    def __call__(self, data):

        return [self.lookup[x] if x in self.lookup else self.default for x in data]


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):

        return torch.from_numpy(data)

    

