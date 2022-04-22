# ------------------------------------------------------------------------------
#
# Project: pytdml
#
# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
import re

import numpy as np
import geojson
from typing import Iterable


def json_empty(item):
    """
    Check if a json item is empty
    Only null, empty collections and empty strings are considered "json empty"
    """
    if item is None:
        return True
    elif isinstance(item, Iterable):
        return not item
    elif isinstance(item, str):
        return item == ""
    else:
        return False


def remove_empty(item):
    """
    Remove empty items from a json item
    """
    if isinstance(item, dict):
        new_item = {k: remove_empty(v) for k, v in item.items()}
        return {k: v for k, v in new_item.items() if not json_empty(v)}
    elif isinstance(item, (list, tuple)):
        new_item = [remove_empty(v) for v in item]
        return [v for v in new_item if not json_empty(v)]
    else:
        return item


def get_bounding_box(geometry):
    """
    Get bbox from geojson geometry object
    """
    coords = np.array(list(geojson.utils.coords(geometry)))
    return [coords[:, 0].min(), coords[:, 1].min(), coords[:, 0].max(), coords[:, 1].max()]


def class_to_index(class_map: dict):
    index = 1
    color_to_index = {}
    if len(class_map) == 2:
        for class_ in class_map:
            findall = re.findall(r'[(](.*?)[)]', class_map[class_])
            color_to_index[eval('(' + findall[0] + ')')[0]] = index
            index = index + 1
    else:
        for class_ in class_map:
            findall = re.findall(r'[(](.*?)[)]', class_map[class_])
            color_to_index[eval('(' + findall[0] + ')')] = index
            index = index + 1
    return color_to_index


def label_to_index(label, color_to_index):
    if len(label.shape) == 3:
        return label_to_index_image(label, color_to_index)
    elif len(label.shape) == 2:
        return gray_to_index_image(label, color_to_index)


def label_to_index_image(label, color_to_index):
    """
    transform 3 channel RGB label image to 1 channel index label image
    """
    assert len(label.shape) == 3
    height, width, channel = label.shape
    index_mat = np.zeros((height, width), dtype=np.uint)
    for rgb_value, class_id in color_to_index.items():
        r_, g_, b_ = rgb_value
        mask_int = (label[:, :, 0] == r_).astype(int) + \
                   (label[:, :, 1] == g_).astype(int) + \
                   (label[:, :, 2] == b_).astype(
                       int)
        mask = mask_int == 3
        index_mat[mask] = class_id
    return index_mat


def gray_to_index_image(label, color_to_index):
    """
    transform 2 channel gray value label image to 1 channel index label image
    """
    assert len(label.shape) == 2
    height, width = label.shape
    index_mat = np.zeros((height, width), dtype=np.uint)
    for gray_value, class_id in color_to_index.items():
        mask_int = label[:, :] == gray_value
        index_mat[mask_int] = class_id
    return index_mat
