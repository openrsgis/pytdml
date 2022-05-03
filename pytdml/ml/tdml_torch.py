# ------------------------------------------------------------------------------
#
# Project: pytdml
#
# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
import cv2
import numpy as np
import torch
from PIL import Image

import pytdml.utils as utils

from torch.utils.data import Dataset


class TorchEOImageSceneTD(Dataset):
    """
    Torch Dataset for EO image scene classification training dataset
    """

    def __init__(self, td_list, class_map, transform=None):
        self.td_list = td_list
        self.class_map = class_map
        self.transform = transform

    def __len__(self):
        return len(self.td_list)

    def __getitem__(self, item):
        img_path = self.td_list[item].data_url
        img = Image.open(img_path)
        img.load()
        label = self.class_map[self.td_list[item].labels[0].label_class]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class TorchEOImageObjectTD(Dataset):
    """
    Torch Dataset for EO image object detection training dataset
    """

    def __init__(self, td_list, class_map, transform=None):
        self.td_list = td_list
        self.class_map = class_map
        self.transform = transform

    def __len__(self):
        return len(self.td_list)

    def __getitem__(self, index):
        img_path = self.td_list[index].data_url
        img = cv2.imread(img_path)
        img_height, img_width, channels = img.shape
        targets = []
        labels = self.td_list[index].labels
        # transform annotations
        for label in labels:
            class_value_ = self.class_map[label.label_class]
            target_ = utils.get_bounding_box(label.object)
            target_[0] = target_[0] / img_width
            target_[1] = target_[1] / img_height
            target_[2] = target_[2] / img_width
            target_[3] = target_[3] / img_height
            target_.append(class_value_)
            targets.append(target_)
        targets = np.array(targets)
        img = img[:, :, (2, 1, 0)]
        if self.transform is not None:
            img, boxes, labels = self.transform(img, targets[:, :4], targets[:, 4])
            targets = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), targets, img_height, img_width


class TorchEOImageSegmentationTD(Dataset):
    """
    Torch Dataset for EO image Semantic Segmentation training dataset
    """

    def __init__(self, td_list, class_map, transform):
        self.td_list = td_list
        self.class_map = class_map
        self.transform = transform
        self.color_to_index = utils.class_to_index(class_map)

    def __len__(self):
        return len(self.td_list)

    def __getitem__(self, item):
        img_path = self.td_list[item].data_url
        img = Image.open(img_path)
        img.load()
        if self.transform is not None:
            img = self.transform(img)
        label_path = self.td_list[item].labels[0].image_url
        label = cv2.imread(label_path, cv2.COLOR_BGR2RGB)
        index_label = utils.label_to_index(label, self.color_to_index)
        index_label = np.asarray(index_label, dtype=np.int64)
        index_label = torch.from_numpy(index_label)
        return img, index_label


def base_transform(image, size, mean, std):
    x = cv2.resize(image, (size[0], size[1]), interpolation=cv2.INTER_AREA).astype(np.float32)
    x /= 255.
    x -= mean
    x /= std
    return x


class BaseTransform:
    def __init__(self, size=None, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
        if size is None:
            size = [256, 256]
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean, self.std), boxes, labels
