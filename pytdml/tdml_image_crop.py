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
import argparse
import json
import os
import sys
import math
import numpy as np

import cv2
from geojson import Feature, Polygon

from pytdml.io import read_from_json
from pytdml.type import EOTrainingDataset, AI_EOTrainingData, AI_PixelLabel
from pytdml.utils import remove_empty


def td_image_crop(td: EOTrainingDataset, save_tdml_path: str, save_crop_dir: str, sub_size: int):
    td_dict = td.to_dict()
    td_list = td_dict['data']
    new_td_list = []
    index = 0
    for d in td_list:
        label_type = d['labels'][0]['type']
        if label_type == 'PixelLabel':
            image_url = d['dataURL']
            label_url = d['labels'][0]['imageURL']
            image_dir = os.path.join(save_crop_dir, "images")
            label_dir = os.path.join(save_crop_dir, "labels")
            if not os.path.isdir(image_dir):
                os.makedirs(image_dir)
            if not os.path.isdir(label_dir):
                os.makedirs(label_dir)
            crop_image_list = image_crop(image_url, image_dir, sub_size)
            crop_label_list = image_crop(label_url, label_dir, sub_size)
            for crop_image_url, crop_label_url in zip(crop_image_list, crop_label_list):
                new_d = AI_EOTrainingData(
                    id=str(index),
                    labels=[AI_PixelLabel(image_url=crop_label_url)],
                    data_url=crop_image_url,
                )
                index = index + 1
                new_td_list.append(new_d.to_dict())
    td_dict["amountOfTrainingData"] = len(new_td_list)
    td_dict['data'] = new_td_list
    with open(save_tdml_path, "w") as f:
        json.dump(remove_empty(td_dict), f, indent=4)


def image_crop(data_url, save_dir, sub_size):
    """
    image crop form data url
    """
    image = cv2.imread(data_url)
    image_name = os.path.basename(data_url)
    h = image.shape[0]
    w = image.shape[1]
    crop_image_path_list = []
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if h > sub_size & (h > sub_size):
        for i in range(int(w / sub_size)):
            for j in range(int(h / sub_size)):
                crop_image_path = os.path.join(save_dir, str(i) + str(j) + image_name)
                crop_image = image[j * sub_size: sub_size * (j + 1), i * sub_size: sub_size * (i + 1)]
                cv2.imwrite(crop_image_path, crop_image)
                crop_image_path_list.append(crop_image_path)
    return crop_image_path_list


class CropWithImage:
    def __init__(self, crop_size=512, overlap=0.25):
        self.crop_size = crop_size
        self.overlap = overlap

    def __call__(self, img, dir, file_name):
        height, width, channel = img.shape
        # 计算滑动窗口的步长
        stride = int(self.crop_size * (1 - self.overlap))

        # 计算需要填充的大小，使得图片可以被整除成多个crop_size大小的块
        pad_h = math.ceil((stride - (height - self.crop_size) % stride) % stride)
        pad_w = math.ceil((stride - (width - self.crop_size) % stride) % stride)
        # 对图像进行填充
        img_pad = np.pad(img, [(0, pad_h), (0, pad_w), (0, 0)], mode='constant', constant_values=0)
        crop_coords_paths = []

        # 依次对每个crop进行处理
        index = 0
        for y in range(0, height + pad_h - stride, stride):
            for x in range(0, width + pad_w - stride, stride):
                # 计算当前crop的边界
                x1, y1 = x, y
                x2, y2 = x + self.crop_size, y + self.crop_size
                # 对crop进行裁剪
                dot_index = file_name.find(".")
                file_name_crop = file_name[:dot_index] + "_cropped_by_" + str(
                    self.crop_size) + "_" + str(int(y / stride)) + "_" + str(int(x / stride)) + \
                                 file_name[dot_index:]
                crop_image_path = os.path.join(dir, file_name_crop)

                if not os.path.exists(crop_image_path):
                    crop = img_pad[y1:y2, x1:x2, :]
                    crop = crop.astype(np.uint8)
                    cv2.imwrite(crop_image_path, crop)

                crop_coords_paths.append(crop_image_path)
                index += 1
        return crop_coords_paths


class CropWithTargetImage(object):
    def __init__(self, crop_size=512, overlap=0.25, threshold=0.35):
        self.crop_size = crop_size
        self.overlap = overlap
        self.threshold = threshold

    def __call__(self, img, target, dir, file_name):
        height, width, channel = img.shape

        # 计算滑动窗口的步长
        stride = int(self.crop_size * (1 - self.overlap))

        # 计算需要填充的大小，使得图片可以被整除成多个crop_size大小的块
        pad_h = math.ceil((stride - (height - self.crop_size) % stride) % stride)
        pad_w = math.ceil((stride - (width - self.crop_size) % stride) % stride)

        # 对图像进行填充

        img_pad = np.pad(img, [(0, pad_h), (0, pad_w), (0, 0)], mode='constant', constant_values=0)

        crop_coords_paths = []
        targets_crops = []

        # 依次对每个crop进行处理
        for y in range(0, height + pad_h - stride, stride):
            for x in range(0, width + pad_w - stride, stride):
                # 计算当前crop的边界
                x1, y1 = x, y
                x2, y2 = x + self.crop_size, y + self.crop_size

                # 对crop进行裁剪
                dot_index = file_name.find(".")
                file_name_crop = file_name[:dot_index] + "_cropped_by_" + str(self.crop_size) + "_" + str(int(y / stride)) + "_" + str(int(x / stride)) \
                                 + file_name[dot_index:]
                crop_image_path = os.path.join(dir, file_name_crop)
                if not os.path.isdir(dir):
                    os.makedirs(dir)
                if not os.path.exists(crop_image_path):
                    crop = img_pad[y1:y2, x1:x2, :]
                    cv2.imwrite(crop_image_path, crop)

                crop_coords_paths.append(crop_image_path)

                targets = []
                for t in target:

                    # 获取目标框的坐标
                    # xmin, ymin, xmax, ymax = t[:4]
                    coordinates = t.object.geometry.coordinates[0]
                    xmin, ymin = coordinates[0]
                    xmax, ymax = coordinates[2]

                    if xmax - xmin == 0:
                        xmax = xmax + 1
                    if ymax - ymin == 0:
                        ymax = ymax + 1

                    # 判断目标框是否在当前crop中
                    if xmin >= x2 or xmax <= x1 or ymin >= y2 or ymax <= y1:
                        continue

                    # 对目标框进行裁剪，并计算其在crop中的坐标
                    crop_x_min = max(xmin - x1, 0)
                    crop_y_min = max(ymin - y1, 0)
                    crop_x_max = min(xmax - x1, self.crop_size)
                    crop_y_max = min(ymax - y1, self.crop_size)

                    crop_width = crop_x_max - crop_x_min
                    crop_height = crop_y_max - crop_y_min

                    # 计算目标框在crop中的面积占比
                    area_intersection = crop_width * crop_height

                    area_bbox = (xmax - xmin) * (ymax - ymin)
                    area_ratio = area_intersection / area_bbox

                    # 如果目标框的面积占比小于阈值，则舍弃这个目标框
                    if area_ratio < self.threshold:
                        continue

                    # 计算目标框在crop中的相对坐标和大小
                    # x_center_crop = (crop_x_min + crop_x_max) / 2 / self.crop_size
                    # y_center_crop = (crop_y_min + crop_y_max) / 2 / self.crop_size
                    # width_crop = crop_width / self.crop_size
                    # height_crop = crop_height / self.crop_size
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[xmin, ymin], [xmin,ymax], [xmax,ymax], [xmax, ymin], [xmin, ymin]]]
                        },
                        "properties": {
                            "name": ""
                        }
                    }

                    targets.append({
                        "object": feature,
                        "class": t.label_class,
                        "area": area_intersection / area_bbox,
                        "isNegative": t.is_negative,
                        "bboxType": t.bbox_type
                    })

                if targets:
                    targets_crops.append(targets)
                else:
                    feature_zero = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0, 0], [0, 0], [0, 0], [0, 0]]]
                        },
                        "properties": {
                            "name": ""
                        }
                    }
                    targets_crops.append([{
                        "object": feature_zero,
                        "class": "",
                        "area": "",
                        "isNegative": "",
                        "isDiffDetectable": "",
                        "bboxType": ""
                    }])
                write_object_label(dir, targets_crops)
        return crop_coords_paths, targets_crops


def write_object_label(dir, target):
    pass


def main():
    parser = argparse.ArgumentParser(description='Encode a training dataset to TrainingDML-AI JSON format based on '
                                                 'YAML configuration file')
    parser.add_argument('--input', type=str, required=True, help='Input original TrainingDML-AU file path')
    parser.add_argument('--output_json', type=str, required=True, help='Output result TrainingDML-AI JSON file path')
    parser.add_argument('--output_images', type=str, required=True, help='Output dir of result cropped images')
    parser.add_argument('--size', type=int, required=True, help='Crop size of images')

    args = parser.parse_args()
    tdml_path = args.input
    save_tdml_path = args.output_json
    save_crop_dir = args.output_images
    sub_size = args.size
    training_datasets = read_from_json(tdml_path)
    if training_datasets:
        td_image_crop(training_datasets, save_tdml_path, save_crop_dir, sub_size)


if __name__ == '__main__':
    sys.exit(main())
