# # ------------------------------------------------------------------------------
# #
# # Project: pytdml
# # Authors: Boyi Shangguan, Kaixuan Wang
# # Created: 2022-05-04
# # Email: sgby@whu.edu.cn
# #
# # ------------------------------------------------------------------------------
# #
# # Copyright (c) 2022 OGC Training Data Markup Language for AI Standard Working Group
# #
# # Permission is hereby granted, free of charge, to any person obtaining a copy
# # of this software and associated documentation files (the "Software"), to deal
# # in the Software without restriction, including without limitation the rights
# # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# # copies of the Software, and to permit persons to whom the Software is
# # furnished to do so, subject to the following conditions:
# #
# # The above copyright notice and this permission notice shall be included in all
# # copies or substantial portions of the Software.
# #
# # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# # SOFTWARE.
# #
# # ------------------------------------------------------------------------------
# import re
# from typing import Iterable
# import json
# import os
# import urllib3
# from PIL import Image
# import numpy as np
# from io import BytesIO
#
#
# from datalibrary.s3Client import minio_client as client
#
#
# def json_empty(item):
#     """
#     Check if a json item is empty
#     Only null, empty collections and empty strings are considered "json empty"
#     """
#     if item is None:
#         return True
#     elif isinstance(item, Iterable):
#         return not item
#     elif isinstance(item, str):
#         return item == ""
#     else:
#         return False
#
#
# def remove_empty(item):
#     """
#     Remove empty items from a json item
#     """
#     if isinstance(item, dict):
#         new_item = {k: remove_empty(v) for k, v in item.items()}
#         return {k: v for k, v in new_item.items() if not json_empty(v)}
#     elif isinstance(item, (list, tuple)):
#         new_item = [remove_empty(v) for v in item]
#         return [v for v in new_item if not json_empty(v)]
#     else:
#         return item
#
#
# def get_bounding_box(geometry):
#     """
#     Get bbox from geojson geometry object
#     """
#
#     coords = np.array(list(geometry["bbox"]))
#     return [coords[0], coords[1], coords[2], coords[3]]
#
#
# def class_to_index(class_map: dict):
#     index = 1
#     color_to_index = {}
#     if len(class_map) == 2:
#         for class_ in class_map:
#             findall = re.findall(r'[(](.*?)[)]', class_map[class_])
#             color_to_index[eval('(' + findall[0] + ')')[0]] = index
#             index = index + 1
#     else:
#         for class_ in class_map:
#             findall = re.findall(r'[(](.*?)[)]', class_map[class_])
#             color_to_index[eval('(' + findall[0] + ')')] = index
#             index = index + 1
#     return color_to_index
#
#
# def label_to_index(label, color_to_index):
#     if len(label.shape) == 3:
#         return label_to_index_image(label, color_to_index)
#     elif len(label.shape) == 2:
#         return gray_to_index_image(label, color_to_index)
#
#
# def label_to_index_image(label, color_to_index):
#     """
#     transform 3 channel RGB label image to 1 channel index label image
#     """
#     assert len(label.shape) == 3
#     height, width, channel = label.shape
#     index_mat = np.zeros((height, width), dtype=np.uint)
#     for rgb_value, class_id in color_to_index.items():
#         r_, g_, b_ = rgb_value
#         mask_int = (label[:, :, 0] == r_).astype(int) + \
#                    (label[:, :, 1] == g_).astype(int) + \
#                    (label[:, :, 2] == b_).astype(int)
#         mask = mask_int == 3
#         index_mat[mask] = class_id
#     return index_mat
#
#
# def gray_to_index_image(label, color_to_index):
#     """
#     transform 2 channel gray value label image to 1 channel index label image
#     """
#     assert len(label.shape) == 2
#     height, width = label.shape
#     index_mat = np.zeros((height, width), dtype=np.uint)
#     for gray_value, class_id in color_to_index.items():
#         mask_int = label[:, :] == gray_value
#         index_mat[mask_int] = class_id
#     return index_mat
#
#
# """
#     luojiaSet datasource functions
# """
#
#
# def get_label_pixel_list_(label_path):
#     """
#         get pxiel value of label pixel class
#     """
#     img = Image.open(label_path)
#     pixels = img.getdata()
#     pixel_list = list(set(pixels))
#     """
#         remove background pixel: 0
#     """
#     try:
#         index = pixel_list.index(0)
#         pixel_list.pop(index)
#         return pixel_list
#     except ValueError:
#         return pixel_list
#
#
# def get_mapping_(file):
#     mapping_data = client.get_object("pytdml", "mapping/" + file + "_mapping.json")
#     map = json.load(BytesIO(mapping_data.read()))
#     return map
#
#
# pixel_map = get_mapping_("pixel")
# name_map = get_mapping_("name")
#
#
# def label_class_list_(pixel_list):
#     """
#         transform label pixel list to label class list
#     """
#     if len(pixel_list) > 0:
#
#         label_class_list = [item["type"] for item in pixel_map if int(item["pngValue"]) in pixel_list]
#         return label_class_list
#     else:
#         return []
#
#
# def object_path_parse_(object_name):
#
#     bucket_name = object_name.split("/")[0]
#     obs_dataset_name = object_name.split("/")[1]
#     dataset_name = dataset_name_map_(obs_dataset_name)
#     semantic_name = object_name.split("/")[2]
#     file_name = object_name.split("/")[3]
#
#     return dataset_name, semantic_name, file_name
#
#
# def generate_file_path(root, url):
#     dataset_name, semantic_name, file_name = object_path_parse_(url)
#     return os.path.join(root, "EOTrainingDataset", dataset_name, semantic_name, file_name)
#
#
# def classList_for_segmentation_(data_item, root):
#     nameList = object_path_parse_(data_item.labels[0].image_url)
#     label_pixel_list = get_label_pixel_list_(
#         os.path.join(root, "EOTrainingDataset", nameList[1], nameList[2], nameList[3]))
#     print(label_pixel_list)
#     label_class_list = label_class_list_(label_pixel_list)
#
#     return label_class_list
#
#
# def regenerate_png_label_(label_array: Image, cls_list):
#     """
#     Zero the pixels in the png label that are not in the category offered by the user
#     """
#     pixel_map = get_mapping_("pixel")
#
#     pixel_list = [int(item["pngValue"]) for item in pixel_map if item["type"] in cls_list]
#
#     mask = np.isin(label_array, pixel_list)
#     label_array[~mask] = 0
#     return label_array
#
#
# def load_data_list_(labels, class_list):
#     for label in labels:
#         if label.label_class in class_list:
#             return True
#     return False
#
#
# def convert_grey_to_rgb_(img):
#     rgb_img = Image.merge('RGB', [img, img, img])
#     return rgb_img
#
#
# def dataset_name_map_(obs_name):
#     """
#     convert obs path name to dataset name
#     """
#     return [d for d in name_map if d["obs_path"] == obs_name][0]["name"]
#
#
# def get_object_label_data_(label, class_map, img_width, img_height):
#     class_value_ = class_map[label.label_class]
#     target_ = get_bounding_box(label.object)
#     target_[0] = target_[0] / img_width
#     target_[1] = target_[1] / img_height
#     target_[2] = target_[2] / img_width
#     target_[3] = target_[3] / img_height
#     target_.append(class_value_)
#     return target_
#
#
# # def collate_fn(batch):
# #     batch = list(zip(*batch))
# #     # batch[0] = nested_tensor_from_tensor_list(batch[0])
# #     return tuple(batch)
# import torch
#
#
# def collate_fn(data):
#     # 将 data 中的图像和标签分别组成两个列表
#     images = [item[0] for item in data]
#     labels = [item[1] for item in data]
#     # 将图像列表转换为 PyTorch 张量，并在第一维上增加一个维度
#     images = torch.stack(images, dim=0)
#     # 将标签列表转换为 PyTorch 张量
#     labels = torch.tensor(labels)
#     return images, labels
#
#
# def datasets_list(dataset_descriptions, task_type, cls_list):
#     if task_type is None:
#         return dataset_descriptions
#     dataset_list = list(filter(lambda item: item['task'] == task_type, dataset_descriptions))
#     if cls_list is None:
#         return dataset_list
#     return [item for item in dataset_list if set(cls_list).issubset(item["classes"])]
#
#
# class LibraryNotInstalledError(Exception):
#     pass
#
#
# def check_object_path(path):
#     object_item = path.split("/")
#     bucket_list = ["scene-classification", "object-detection", "land-cover", "change-detection", "3d-construction"]
#     if len(object_item) == 4 and object_item[0] in bucket_list:
#         return True
#     return False
#
#
# def transform_annotation(labels, class_map, img_width, img_height):
#     # transform annotations
#     targets = []
#     for label in labels:
#         target_ = get_object_label_data_(label, class_map, img_width, img_height)
#         targets.append(target_)
#
#     targets = np.array(targets)
#     return targets
#
#
# def image_open(data):
#     """
#     读取影像，如果是高光谱影像,则使用rasterio来读取
#     """
#     try:
#
#         with Image.open(data) as img:
#             np_img = np.asarray(img).copy()
#
#         if len(np_img.shape) == 2:
#             np_img = np.expand_dims(np_img, axis=2)
#         # return np.transpose(np_img, (2, 0, 1))
#         return np_img
#     except IOError:
#         try:
#             import rasterio
#         except ModuleNotFoundError:
#             raise LibraryNotInstalledError("failed to import rasterio, please install the library first")
#         if isinstance(data, str):
#             pass
#         elif isinstance(data, urllib3.response.HTTPResponse):
#             data = BytesIO(data.read())
#         else:
#             raise ValueError("failed to read image")
#         with rasterio.open(data) as image:
#             img = image.read().copy()
#         return img
