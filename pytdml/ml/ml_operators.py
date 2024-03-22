# ------------------------------------------------------------------------------
#
# Project: pytdml
# Authors: Boyi Shangguan, Kaixuan Wang
# Created: 2022-05-04
# Email: sgby@whu.edu.cn
#
# ------------------------------------------------------------------------------
#
# Copyright (c) 2022 OGC Training Data Markup Language for AI Standard Working Group
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ------------------------------------------------------------------------------
import random

import torch

from pytdml.type import TrainingDataset


def split_train_valid_test(td: TrainingDataset, train_ratio=0.7, valid_ratio=0.0, test_ratio=0.3):
    """
    Splits a TrainingDataset into train, valid and test sets.
    """
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) <= 0.000001, "Ratios must sum to 1.0"

    n_samples = len(td.data)
    n_train = int(n_samples * train_ratio)
    n_valid = int(n_samples * valid_ratio)

    random.shuffle(td.data)

    train_td = td.data[:n_train]
    valid_td = td.data[n_train:n_train + n_valid]
    test_td = td.data[n_train + n_valid:]

    return train_td, valid_td, test_td


def create_class_map(td: TrainingDataset):
    """
    Creates a map from class labels to indices.
    """
    assert td.classes is not None, "Training dataset has no classes"
    class_map = {}
    for i, _class in enumerate(td.classes):
        if isinstance(_class, dict):
            for item in _class.items():
                class_map[item[0]] = item[1]
        else:
            class_map[_class] = i
    return class_map


def create_classes_map_(classes):
    """
    Creates a map from class labels to indices for luojiaSet.
    """
    assert classes is not None, "Training dataset has no classes"
    class_map = {}
    for i, _class in enumerate(classes):
        if isinstance(_class, dict):
            for item in _class.items():
                class_map[item[0]] = item[1]
        else:
            class_map[_class] = i
    return class_map


def collate_fn(batch):
    img, targets = list(zip(*batch))
    return torch.stack(img, 0), list(targets)

