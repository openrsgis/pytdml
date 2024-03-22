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
import cv2
import numpy as np
import torch
from PIL import Image
import os

import pytdml.utils as utils

from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset


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


class TorchSceneClassificationTD(VisionDataset):
    """
    Torch Dataset for EO image scene classification training dataset
    """

    def __init__(self, td_list, root, class_map, transform=None):
        super().__init__(root=root, transform=transform)
        self._basedir = root

        self.td_list = td_list
        self.imgs, self.labels = self._load_img_label()

        self.class_map = class_map

    def _load_img_label(self):
        # self.imgs = [utils.generate_file_path(self._basedir, self.root) for item in self.td_list]
        # imgs = [os.path.join(self._basedir, "EOTrainingDataset", self.tdml.name, class_name, file)
        #         for class_name in self.tdml.classes
        #         for file in os.listdir(os.path.join(self._basedir, "EOTrainingDataset", self.tdml.name, class_name))]
        # imgs = [os.path.join(self._basedir, "EOTrainingDataset", *utils.object_path_parse_(item.data_url))
        #         for item in self.td_list]
        imgs = [item.data_url[0] for item in self.td_list]
        labels = [item.labels[0].label_class for item in self.td_list]
        return imgs, labels

    def __len__(self):
        return len(self.td_list)

    def __getitem__(self, item):

        img_path = self.imgs[item]
        img = utils.image_open(img_path)
        # single band check
        img = utils.channel_processing(img)

        label = self.class_map[self.td_list[item].labels[0].label_class]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def class_to_idx(self):
        return self.class_map


class TorchObjectDetectionTD(VisionDataset):
    """
    Torch Dataset for EO image object detection training dataset
    """

    def __init__(self, td_list, root, class_map, transform=None):
        super().__init__(root=root, transform=transform)
        self._basedir = root

        self.td_data_list = td_list
        self.class_map = class_map

        self.images = [item.data_url[0] for item in self.td_data_list]
        self.transform = transform

    def __len__(self):
        return len(self.td_data_list)

    def __getitem__(self, index):

        img = utils.image_open(self.images[index])

        img_height, img_width, channel = img.shape
        img = utils.channel_processing(img)
        labels = self.td_data_list[index].labels
        targets = utils.target_to_dict(labels, self.class_map, img_width, img_height)

        if self.transform is not None:
            img, targets = self.transform(img, targets)
        return img, targets


class TorchSemanticSegmentationTD(VisionDataset):
    """
    Torch Dataset for EO image Semantic Segmentation training dataset
    """

    def __init__(self, td_list, root, classes, transform=None):
        super(TorchSemanticSegmentationTD, self).__init__(None, transform=transform)
        self._basedir = root
        self.class_list = classes
        self.transform = transform

        self.td_data_list = self._load_data(td_list)
        self._imgs, self._labels = self._load_img_label(self.td_data_list)

    def __len__(self):
        return len(self.td_data_list)

    def _load_data(self, td_list):
        """
        """
        return td_list

    def _load_img_label(self, td_list):

        # for item in td_list:
        #     img_path = utils.generate_file_path(self._basedir, item.data_url)
        #     img_paths.append(img_path)
        #
        #     label_path = item.labels[0].image_url
        #     label_path = utils.generate_file_path(self._basedir, label_path)
        #     label_paths.append(label_path)
        img_paths = []
        label_paths = []
        for item in td_list:
            # if utils.check_object_path(item.data_url[0]) and utils.check_object_path(item.labels[0].image_url):
            #     label_paths.append(os.path.join(self._basedir, "EOTrainingDataset",
            #                                     *utils.object_path_parse_(item.labels[0].image_url)))
            #     img_paths.append(os.path.join(self._basedir, "EOTrainingDataset",
            #                                   *utils.object_path_parse_(item.data_url)))
            # else:
            #     img_paths.append(item.data_url[0])
            #     label_paths.append(item.labels[0].image_url)
            img_paths.append(item.data_url[0])
            label_paths.append(item.labels[0].image_url)
        return img_paths, label_paths

    def __getitem__(self, item):
        image = utils.image_open(self._imgs[item])

        # image = torch.from_numpy(image).float().contiguous()

        label = utils.image_open(self._labels[item])

        # scheme 2
        if self.class_list is not None:
            label = utils.regenerate_png_label_(label, self.class_list)
        # label = torch.from_numpy(label).long().squeeze()

        # label = np.array(label)

        #
        # label = torch.from_numpy(label).long()
        # label = label.squeeze()

        if self.transform is not None:
            image = self.transform(image)
            label = self.transform(label)
        return image, label


class TorchChangeDetectionTD(VisionDataset):
    def __init__(self, td_list, root, transform):
        super().__init__(root=root, transform=transform)

        self._basedir = root

        self.td_list = td_list

        self.transform = transform

        self._sample = self._load_sample()

    def __len__(self):
        return len(self.td_list)

    def _load_sample(self):
        sample = []
        for item in self.td_list:
            data_url = item.data_url
            label_url = item.labels[0].image_url
            # if utils.check_object_path(data_url[0]):
            #     before_img_path = utils.generate_local_file_path(self._basedir, data_url[0])
            #     after_img_path = utils.generate_local_file_path(self._basedir, data_url[1])
            #     label_path = utils.generate_local_file_path(self._basedir, label_url)
            #     sample.append([before_img_path, after_img_path, label_path])
            # else:
            #     sample.append([data_url[0], data_url[1], label_url])
            sample.append([data_url[0], data_url[1], label_url])

        return sample

    def __getitem__(self, item):
        img_before_path, img_after_path, label_path = self._sample[item]
        before_img = utils.image_open(img_before_path)
        after_img = utils.image_open(img_after_path)
        before_img = utils.channel_processing(before_img)
        after_img = utils.channel_processing(after_img)
        label = utils.image_open(label_path)
        # label = np.array(label)

        # label = torch.from_numpy(label).long()
        # label = label.squeeze()

        if self.transform is not None:
            before_img = self.transform(before_img)
            after_img = self.transform(after_img)
            label = self.transform(label)
        return before_img, after_img, label


class TorchStereoTD(VisionDataset):
    def __init__(self, td_list, root, transform=None):
        super().__init__(root)
        self.root = root

        self.td_list = td_list
        self.transform = transform
        self.target_imgs, self.ref_imgs, self.disp_imgs = self._load_data()  # 加载相机参数

    def _load_data(self):

        target_imgs = [item.data_url[0] for item in self.td_list]
        ref_imgs = [item.data_url[1] for item in self.td_list]
        disp_imgs = [item.labels[0].image_url for item in self.td_list]
        return target_imgs, ref_imgs, disp_imgs

    def __len__(self):
        return len(self.td_list)

    def __getitem__(self, item):
        disp_img = utils.image_open(self.disp_imgs[item])
        target_img = utils.image_open(self.target_imgs[item])
        ref_img = utils.image_open(self.ref_imgs[item])

        if self.transform is not None:
            target_img = self.transform(target_img)
            ref_img = self.transform(ref_img)
            disp_img = self.transform(disp_img)
        return target_img, ref_img, disp_img


class Torch3DModelReconstructionTD(VisionDataset):
    def __init__(self, tdml, root, transform=None):
        super().__init__(root)
        self.root = root
        self.cams, self.depths, self.images = self._load_data()   # 加载相机参数
        self.tdml = tdml
        self.transform = transform

    def _load_data(self):
        cams_dir = os.path.join(self.root, self.tdml.name, "Cams")
        cams = os.listdir(cams_dir)

        depths_dir = os.path.join(self.root, self.tdml.name, "Depths")
        depths = os.listdir(depths_dir)

        imgs_dir = os.path.join(self.root, self.tdml.name, "Images")
        imgs = [os.listdir(item) for item in os.listdir(imgs_dir)]
        imgs = list(zip(*imgs))

        return cams, depths, imgs

    def __len__(self):

        return len(self.tdml.data)

    def __getitem__(self, item):
        cam = self.cams[item]
        depth = self.depths[item]
        image_list = self.images[item]  # 对于每个样本，使用不同的视角图像，即取模操作
        images = [utils.image_open(i) for i in image_list]
        depth_imgs = [utils.image_open(i) for i in depth]
        cam_txt = []
        for i in cam:
            with open(i) as f:
                lines = f.readlines()
                cam_txt.append(lines)
        if self.transform is not None:
            images = [self.transform(image) for image in images]
        return images, depth_imgs, cam_txt


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
