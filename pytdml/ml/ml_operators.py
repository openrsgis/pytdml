# ------------------------------------------------------------------------------
#
# Project: pytdml
#
# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
import random

from pytdml import TrainingDataset


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


def creat_class_map(td: TrainingDataset):
    """
    Creates a map from class labels to indices.
    """
    assert td.classes is not None, "Training dataset has no classes"
    class_map = {}
    for i, _class in enumerate(td.classes):
        class_map[_class] = i
    return class_map
