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
import pickle
from typing import Iterable
import json
import os
import urllib3
from PIL import Image
from rasterio import RasterioIOError

Image.MAX_IMAGE_PIXELS = 10_000_000_000  # 10 billion

import numpy as np
from io import BytesIO
import re
import torch

from datalibrary.s3Client import minio_client as client


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
    print(geometry)
    coords = np.array(list(geometry["geometry"]["coordinates"][0]))
    return [coords[0][0], coords[0][1], coords[2][0], coords[3][1]]


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
                   (label[:, :, 2] == b_).astype(int)
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


"""
    luojiaSet datasource functions
"""


def get_label_pixel_list_(label_path):
    """
        get pxiel value of label pixel class
    """
    img = Image.open(label_path)
    pixels = img.getdata()
    pixel_list = list(set(pixels))
    """
        remove background pixel: 0
    """
    try:
        index = pixel_list.index(0)
        pixel_list.pop(index)
        return pixel_list
    except ValueError:
        return pixel_list


def get_mapping_(file):
    mapping_data = client.get_object("pytdml", "mapping/" + file + "_mapping.json")
    map = json.load(BytesIO(mapping_data.read()))
    return map


pixel_map = get_mapping_("pixel")
name_map = get_mapping_("name")


def label_class_list_(pixel_list):
    """
        transform label pixel list to label class list
    """
    if len(pixel_list) > 0:

        label_class_list = [item["type"] for item in pixel_map if int(item["pngValue"]) in pixel_list]
        return label_class_list
    else:
        return []


def split_data_url(data_url):
    """
       Splits the data URL into bucket name and object name.

       Args:
           data_url (str): The data URL to split.

       Returns:
           tuple: The bucket name and object name.

       """
    bucket_name, object_name = data_url.split("/", 1)
    return bucket_name, object_name


def object_path_parse_(object_name):
    # bucket_name = object_name.split("/")[0]
    # obs_dataset_name = object_name.split("/")[1]
    # dataset_name = dataset_name_map_(obs_dataset_name)
    # semantic_name = object_name.split("/")[2]
    # file_name = object_name.split("/")[3]
    name_list = object_name.split("/")
    name_list[1] = dataset_name_map_(name_list[1])
    name_list.pop(0)

    return name_list


def generate_local_file_path(root, data_url):
    name_list = object_path_parse_(data_url)
    return os.path.join(root, "EOTrainingDataset", *name_list)


def classList_for_segmentation_(data_item, root):
    nameList = object_path_parse_(data_item.labels[0].image_url)
    label_pixel_list = get_label_pixel_list_(
        os.path.join(root, "EOTrainingDataset", nameList[1], nameList[2], nameList[3]))
    print(label_pixel_list)
    label_class_list = label_class_list_(label_pixel_list)

    return label_class_list


def regenerate_png_label_(label_array: Image, cls_list):
    """
    Zero the pixels in the png label that are not in the category offered by the user
    """
    pixel_map = get_mapping_("pixel")

    pixel_list = [int(item["pngValue"]) for item in pixel_map if item["type"] in cls_list]

    mask = np.isin(label_array, pixel_list)
    label_array[~mask] = 0
    return label_array


def load_data_list_(labels, class_list):
    for label in labels:
        if label.label_class in class_list:
            return True
    return False


def convert_grey_to_rgb_(img):
    rgb_img = np.repeat(img, 3, axis=2)
    return rgb_img


def dataset_name_map_(obs_name):
    """
    convert obs path name to dataset name
    """
    return [d for d in name_map if d["obs_path"] == obs_name][0]["name"]


def get_object_label_data_(label, img_width, img_height):
    target_ = get_bounding_box(label.object)
    target_[0] = target_[0] / img_width
    target_[1] = target_[1] / img_height
    target_[2] = target_[2] / img_width
    target_[3] = target_[3] / img_height
    return target_


def generate_new_tdml(dataset, classes):
    new_id = '_'.join([ds.id for ds in dataset])
    new_name = '&'.join([ds.name for ds in dataset])
    new_description = "samples of " + str(classes) + " from datasets: " + new_name
    new_image_size = list(set([ds.image_size for ds in dataset]))
    new_bands = list(set([elem for ds in dataset for elem in ds.bands]))
    return new_id, new_name, new_description, new_image_size, new_bands


def datasets_list(dataset_descriptions, task_type, cls_list):
    if task_type is None:
        return dataset_descriptions
    dataset_list = list(filter(lambda item: item['task'] == task_type, dataset_descriptions))
    if cls_list is None:
        return dataset_list
    return [item for item in dataset_list if set(cls_list).issubset(item["classes"])]


class LibraryNotInstalledError(Exception):
    pass


def check_object_path(path):
    object_item = path.split("/")
    bucket_list = ["scene-classification", "object-detection", "land-cover", "change-detection", "3d-construction"]
    if len(object_item) == 4 and object_item[0] in bucket_list:
        return True
    return False


def transform_annotation(labels, class_map, img_width, img_height):
    # transform annotations
    targets = []
    for label in labels:
        target_ = get_object_label_data_(label, img_width, img_height)
        targets.append(target_)

    targets = np.array(targets)
    return targets


def generate_cache_file_path(root, name, crop=None):
    """
        Generates a cache file path based on the given name and crop parameters.

        Args:
            name (str): The base name for the cache file.
            crop (tuple or None): The crop parameters.
            root (str): The root directory.

        Returns:
            str: The generated cache file path.

        """
    cache_name = name + ".pkl"
    if crop is not None:
        crop_str = '_'.join(str(i) for i in crop)
        cache_name = name + "_" + crop_str + ".pkl"
    cache_file_path = os.path.join(root, "EOTrainingDataset", ".cache", cache_name)
    if not os.path.exists(os.path.dirname(cache_file_path)):
        os.makedirs(os.path.dirname(cache_file_path))
    return cache_file_path


def target_to_dict(labels, class_map, img_width, img_height):
    target_dict = {"bbox": [], "class": [], "bboxType": [], "isDifficultlyDetectable": [], "isNegative": []}
    for label in labels:
        normalized_target = get_object_label_data_(label, img_width, img_height)
        target_dict["bbox"].append(normalized_target)

        if label.label_class == "":
            target_dict["class"].append(torch.tensor(-1))
        else:
            if label.label_class in class_map:
                target_dict["class"].append((torch.tensor(class_map[label.label_class], dtype=torch.int64)))
        target_dict["bboxType"].append(label.bbox_type)
        target_dict["isNegative"].append(label.is_negative)
    target_dict["bbox"] = torch.tensor(target_dict["bbox"])
    target_dict["class"] = torch.tensor(target_dict["class"])
    return target_dict


def cache_dump(cache_file_path, root, td_list):
    if not os.path.exists(os.path.dirname(cache_file_path)):
        os.makedirs(os.path.dirname(cache_file_path))
    with open(os.path.join(root, cache_file_path), "wb") as f:
        pickle.dump(td_list, f)


def load_cached_training_data(cache_file_path):
    """
        Loads cached training data from the specified cache file.

        Args:
            cache_file_path (str): The path to the cache file.

        Returns:
            list: The loaded training data list.

    """
    td_list = []
    if os.path.exists(cache_file_path):
        with open(cache_file_path, "rb") as f:
            td_list = pickle.load(f)
    return td_list


def channel_processing(img):
    img_height, img_width, channel = img.shape

    # single band check
    if channel == 1:
        img = convert_grey_to_rgb_(img)
    # TODO: 通道数处理
    if channel > 3:
        img = img[:, :, :3]
    return img


def image_open(data):
    """
    Reads an image from a file or a HTTPResponse object.
    If the image is a hyper spectral image, uses rasterio to read it.

    Args:
        data: The file path or a HTTPResponse object containing the image data.

    Returns:
        A numpy array representing the image data.
        For grayscale images, the array has shape (height, width, 1). For color
        images, the array has shape (height, width, channel).

    Raises:
        LibraryNotInstalledError: If the rasterio library is not installed.
        ValueError: If the image cannot be read.
    """

    try:
        with Image.open(data) as img:

            if img.mode == '1':  # Handling the case of bit depth of 1
                img = img.convert("L")
            np_img = np.array(img, dtype=np.float64).copy()

        if len(np_img.shape) == 2:
            np_img = np.expand_dims(np_img, axis=2)

        return np_img
    except IOError:
        try:
            import rasterio
        except ModuleNotFoundError:
            raise LibraryNotInstalledError("failed to import rasterio, please install the library first")
        if isinstance(data, str):
            pass
        elif isinstance(data, urllib3.response.HTTPResponse):
            data = BytesIO(data.read())
        else:
            raise ValueError("failed to read image")
        try:
            with rasterio.open(data) as image:
                img = image.read().copy()
            return img
        except (rasterio.errors.RasterioIOError, Exception) as e:
            raise ValueError("Failed to read image with both PIL and rasterio libraries. "
                             "Please ensure you have downloaded the dataset correctly "
                             "or check the file path or URL.")


def save_cache(cache_path, cache_file_list):
    if not os.path.exists(cache_path):
        if not os.path.exists(os.path.dirname(cache_path)):
            os.makedirs(os.path.dirname(cache_path))
        with open(cache_path, "wb") as f:
            pickle.dump(cache_file_list, f)


def parse_s3_path(s3_path):
    # Match and extract the bucket and key using regular expressions
    pattern = r'^s3://([^/]+)/(.+)$'
    match = re.match(pattern, s3_path)
    if match:
        bucket_name = match.group(1)
        key = match.group(2)
        return bucket_name, key
    else:
        raise ValueError('Invalid S3 path')


def is_s3_path(path):
    pattern = r'^s3://.+/.+$'
    return re.match(pattern, path) is not None


def s3_path_to_data(user_client, data_url):
    bucket_name, key = parse_s3_path(data_url)
    byte_data = user_client.get_object(bucket_name, key)["Body"].read()
    return BytesIO(byte_data)


import random


def split_data(dataset, split_type=None, split_ratio=None):
    td_data = dataset.data
    if split_type and split_ratio:
        raise ValueError("Only one of 'split_type' or 'split_ratio' should be provided.")

    if split_type:
        if split_type == 'training':

            td_data = [data for data in td_data if data.trainingType == "training"]
            if len(td_data) == 0:
                raise ValueError("The original dataset training type record is missing, please use ratio division.")
        elif split_type == 'validation':
            td_data = [data for data in td_data if data.trainingType == "validation"]
            if len(td_data) == 0:
                raise ValueError("The original dataset training type record is missing, please use ratio division.")
        elif split_type == 'test':
            td_data = [data for data in td_data if data.trainingType == "test"]
            if len(td_data) == 0:
                raise ValueError("The original dataset training type record is missing, please use ratio division.")
        else:
            raise ValueError("Invalid value for 'split_type'.")
        dataset.data = td_data
        return dataset

    if split_ratio:
        assert split_ratio[0] + split_ratio[1] + split_ratio[2] == 1.0
        total_count = len(td_data)
        train_count = int(split_ratio[0] * total_count)
        validation_count = int(split_ratio[1] * total_count)
        test_count = total_count - train_count - validation_count

        if train_count < 0 or validation_count < 0 or test_count < 0:
            raise ValueError("Split ratios should be non-negative and sum up to 1.")

        shuffled_data = random.sample(td_data, total_count)
        # 划分数据集
        train_set = shuffled_data[:train_count]
        valid_set = shuffled_data[train_count:train_count + validation_count]
        test_set = shuffled_data[train_count + validation_count:]

        # 返回划分后的三个数据集
        dataset.data = train_set
        data1 = dataset
        dataset.data = valid_set
        data2 = dataset
        dataset.data = test_set
        data3 = dataset

        return data1, data2, data3


    return td_data  # Return original data if no split type or ratio is provided
