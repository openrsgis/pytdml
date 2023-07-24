# ------------------------------------------------------------------------------
#
# Project: pytdml
# Authors: Shuaiqi Liu
# Created: 2023-02-22
# Email: sqi_liu@whu.edu.cn
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
import glob
import json
import pickle
from abc import ABC

import cv2
import geojson
import numpy as np
import torch
import io
from PIL import Image
import os
import math


import pytdml.utils as utils

from torch.utils.data import Dataset

from torchdata.datapipes.iter import IterDataPipe

from datalibrary import downloader
from datalibrary.s3Client import minio_client as client
from pytdml.type import ObjectLabel
from pytdml.utils import image_open, save_cache
from pytdml.tdml_image_crop import CropWithImage, CropWithTargetImage


class TorchSceneClassificationDataPipe(IterDataPipe, ABC):
    def __init__(self, td_list, root, cache_path, class_map, transform=None):

        self.class_map = class_map

        self.td_list = td_list
        self._basedir = root
        self.cache_path = cache_path
        self._cache_file_list = []
        self.transform = transform

    def __iter__(self):
        iterator = worker_load_process(self.td_list)
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as cache_file:
                td_list = pickle.load(cache_file)
            iterator = worker_load_process(td_list)
            for item in iterator:
                img = image_open(item.data_url[0])
                img = utils.channel_processing(img)
                label = self.class_map[item.labels[0].label_class]
                if self.transform is not None:
                    img = self.transform(img)
                yield img, label
        else:
            for item in iterator:
                sample_url = item.data_url[0]

                img, file_path = downloader.download_remote_object(self._basedir, sample_url)
                label = self.class_map[item.labels[0].label_class]
                if self.transform is not None:
                    img = self.transform(img)
                item.data_url = [file_path]
                self._cache_file_list.append(item)
                yield img, label
            save_cache(self.cache_path, self._cache_file_list)


class TorchObjectDetectionDataPipe(IterDataPipe):
    def __init__(self, td_list, root, cache_path, class_map, crop=None, transform=None):
        super().__init__()

        self.td_list = td_list

        self._basedir = root
        self.class_map = class_map
        self.crop = crop
        self.transform = transform
        self._cache_file_list = []
        self.cache_path = cache_path

    def __iter__(self):
        if os.path.exists(self.cache_path):

            with open(self.cache_path, "rb") as cache_file:
                td_list = pickle.load(cache_file)
            iterator = worker_load_process(td_list)
            for item in iterator:
                img = image_open(item.data_url[0])
                img_height, img_width, channel = img.shape
                img = utils.channel_processing(img)
                targets = utils.target_to_dict(item.labels, self.class_map, img_width, img_height)
                if self.transform is not None:
                    img, target = self.transform(img, targets)
                yield img, targets

        else:
            iterator = worker_load_process(self.td_list)
            for item in iterator:
                sample_url = item.data_url[0]

                img, file_path = downloader.download_remote_object(self._basedir, sample_url)

                img_height, img_width, channel = img.shape
                #  band processing
                img = utils.channel_processing(img)

                if self.crop is None:
                    # transform annotations
                    targets = utils.target_to_dict(item.labels, self.class_map, img_width, img_height)
                    if self.transform is not None:
                        img, targets = self.transform(img, targets)
                    item.data_url = [file_path]
                    self._cache_file_list.append(item)
                    yield img, targets

                else:

                    crop_object = CropWithTargetImage(*self.crop)
                    crop_paths, targets = crop_object(img, item.labels, os.path.dirname(file_path),
                                                      sample_url.split("/")[-1])

                    for crop_path, target_ in zip(crop_paths, targets):
                        img = image_open(crop_path)
                        img = utils.channel_processing(img)
                        labels = []
                        for target in target_:

                            json_object = {"bbox": target["bbox"], "type": "Feature"}

                            labels.append(ObjectLabel(object=geojson.loads(json.dumps(json_object)),
                                                      label_class=target["class"],
                                                      bbox_type=target["bboxType"],
                                                      is_negative=target["isNegative"],
                                                      ))
                        target_dict = utils.target_to_dict(labels, self.class_map, img_width, img_height)
                        if self.transform is not None:
                            img, targets = self.transform(img, target_dict)
                        item.data_url = [file_path]
                        item.labels = labels
                        self._cache_file_list.append(item)
                        yield img, targets
            save_cache(self.cache_path, self._cache_file_list)


class TorchSemanticSegmentationDataPipe(IterDataPipe, ABC):

    def __init__(self, td_list, root, cache_path, class_list=None, crop=None, transform=None):
        super().__init__()

        self.class_list = class_list
        self.transform = transform

        self.td_list = td_list
        self._basedir = root
        self.crop = crop
        self.transform = transform
        self._cache_file_list = []
        self.cache_path = cache_path

    def __iter__(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as cache_file:
                td_list = pickle.load(cache_file)
            iterator = worker_load_process(td_list)
            for item in iterator:
                img = image_open(item.data_url[0])
                img = utils.channel_processing(img)
                label = image_open(item.labels[0].image_url)
                if self.transform is not None:
                    img = self.transform(img)
                    label = self.transform(label)
                yield img, label
        else:
            iterator = worker_load_process(self.td_list)

            for item in iterator:

                sample_url = item.data_url[0]

                label_url = item.labels[0].image_url
                img, image_path = downloader.download_remote_object(self._basedir, sample_url)
                img = utils.channel_processing(img)
                label, label_path = downloader.download_remote_object(self._basedir, label_url)
                label = utils.regenerate_png_label_(label, self.class_list)

                if self.crop is None:
                    if self.transform is not None:
                        img = self.transform(img)
                        label = self.transform(label)
                    item.data_url = [image_path]
                    item.labels[0].image_url = label_path
                    self._cache_file_list.append(item)
                    yield img, label
                else:

                    crop_object = CropWithImage(*self.crop)  # 补充参数
                    image_crop_paths = crop_object(img, os.path.dirname(image_path), sample_url.split("/")[-1])
                    label_crop_paths = crop_object(label, os.path.dirname(label_path), label_url.split("/")[-1])

                    for i in range(len(image_crop_paths)):
                        img = image_open(image_crop_paths[i])
                        img = utils.channel_processing(img)
                        label = image_open(label_crop_paths[i])
                        if self.transform is not None:
                            img = self.transform(img)
                            label = self.transform(label)
                        item.data_url = [image_path]
                        item.labels[0].image_url = label_path
                        self._cache_file_list.append(item)
                        yield img, label
            save_cache(self.cache_path, self._cache_file_list)


class TorchChangeDetectionDataPipe(IterDataPipe, ABC):

    def __init__(self, td_list, root, cache_path, crop=None, transform=None):
        super().__init__()

        self.td_list = td_list
        self._basedir = root
        self.crop = crop
        self.transform = transform
        self._cache_file_list = []
        self.cache_path = cache_path

    def __iter__(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as cache_file:
                td_list = pickle.load(cache_file)
            iterator = worker_load_process(td_list)
            for item in iterator:
                bef_img = image_open(item.data_url[0])
                bef_img = utils.channel_processing(bef_img)
                af_img = image_open(item.data_url[1])
                af_img = utils.channel_processing(af_img)
                label = image_open(item.labels[0].image_url)
                if self.transform is not None:
                    bef_img = self.transform(bef_img)
                    af_img = self.transform(af_img)
                    label = self.transform(label)

                yield bef_img, af_img, label
        else:

            iterator = worker_load_process(self.td_list)

            for item in iterator:
                data_url = item.data_url
                label_url = item.labels[0].image_url
                before_img, before_img_path = downloader.download_remote_object(self._basedir, data_url[0])
                after_img, after_img_path = downloader.download_remote_object(self._basedir, data_url[1])
                label, label_path = downloader.download_remote_object(self._basedir, label_url)

                if self.crop is None:
                    if self.transform is not None:
                        before_img = self.transform(before_img)
                        after_img = self.transform(after_img)
                        label = self.transform(label)
                    item.data_url = [before_img_path, after_img_path]
                    item.labels[0].image_url = label_path
                    self._cache_file_list.append(item)
                    yield before_img, after_img, label
                else:
                    after_img = image_open(after_img_path)
                    label = image_open(label_path)

                    crop_object = CropWithImage(*self.crop)  # 补充参数
                    before_image_crop_paths = crop_object(before_img, os.path.dirname(before_img_path),
                                                          data_url[0].split("/")[-1])
                    after_image_crop_paths = crop_object(after_img, os.path.dirname(after_img_path),
                                                          data_url[1].split("/")[-1])
                    label_crop_paths = crop_object(label, os.path.dirname(label_path), label_url.split("/")[-1])
                    for i in range(len(before_image_crop_paths)):
                        before_img = image_open(before_image_crop_paths[i])
                        after_img = image_open(after_image_crop_paths[i])
                        label = image_open(label_crop_paths[i])
                        if self.transform is not None:
                            before_img = self.transform(before_img)
                            after_img = self.transform(after_img)
                            label = self.transform(label)
                        item.data_url = [before_img_path, after_img_path]
                        item.labels[0].image_url = label_path
                        self._cache_file_list.append(item)
                        yield before_img, after_img, label
            save_cache(self.cache_path, self._cache_file_list)


class TorchStereoDataPipe(IterDataPipe, ABC):

    def __init__(self, td_list, root, transform=None):
        super().__init__()

        self.transform = transform

        self.td_list = td_list
        self._basedir = root

        self.transform = transform

    def __iter__(self):
        iterator = worker_load_process(self.td_list)

        for item in iterator:
            sample_url = item.data_url
            samples = []
            for url in sample_url:
                img, file_url = downloader.download_remote_object(self._basedir, url)

                if self.transform is not None:
                    img = self.transform(img)
                samples.append(img)

            label, label_path = downloader.download_remote_object(self._basedir, item.labels[0].image_url)

            yield samples, label


class Torch3DModelConstructionDataPipe(IterDataPipe):

    def __init__(self, td_list, root, transform=None):
        super().__init__()

        self.transform = transform
        self.td_list = td_list
        self._basedir = root

        self.transform = transform

    def __iter__(self):
        iterator = worker_load_process(self.td_list)

        for item in iterator:
            sample_url = item.data_url

            sample_paths = []
            for url in sample_url:
                bucket_name, sample_name = sample_url[0].split(os.path.pathsep, 1)
                sample_path = utils.generate_file_path(self._basedir, url)
                if not os.path.exists(os.path.dirname(sample_path)):
                    os.makedirs(os.path.dirname(sample_path))
                if not os.path.exists(sample_path):
                    client.fget_object(bucket_name, sample_name, sample_path)
                sample_paths.append(sample_path)

            cams_paths = []
            for cam_url in item.cams:
                bucket_name, cams_name = cam_url[0].split(os.path.pathsep, 1)
                cam_path = utils.generate_file_path(self._basedir, cam_url)
                if not os.path.exists(os.path.dirname(cam_path)):
                    os.makedirs(os.path.dirname(cam_path))
                if not os.path.exists(cam_path):
                    client.fget_object(bucket_name, cams_name, cam_path)
                cams_paths.append(cam_path)
            label_paths = []
            for label_url in item.labels.imageUrl:
                bucket_name, label_name = label_url[0].split(os.path.pathsep, 1)
                label_path = utils.generate_file_path(self._basedir, label_url)
                if not os.path.exists(os.path.dirname(label_path)):
                    os.makedirs(os.path.dirname(label_path))
                if not os.path.exists(label_path):
                    client.fget_object(bucket_name, label_name, label_path)
                label_paths.append(label_path)

            cam = np.load(self.cams[item])
            depth = np.load(self.depths[item])
            image_list = self.images[item]  # 对于每个样本，使用不同的视角图像，即取模操作
            images = [Image.open(i).convert("RGB") for i in image_list]
            if self.transform is not None:
                # 如果定义了transform，就对图像进行处理
                images = [self.transform(image) for image in images]
            yield cam, depth, images


def worker_load_process(td_list):
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:  # single-process data loading, return the full iterator
        iterator = td_list

    else:  # in a worker process
        # split workload
        per_worker = int(math.ceil(len(td_list) / float(worker_info.num_workers)))
        worker_id = worker_info.id
        iter_start = worker_id * per_worker
        iter_end = min(iter_start + per_worker, len(td_list))
        iterator = td_list[iter_start:iter_end]
    return iterator



