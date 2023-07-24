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
import tensorflow as tf
import tensorflow_io as tfio

import pytdml.utils as utils
from datalibrary.downloader import *


def _parse_image(filename):
    """
    Parse a single image file.
    """
    file_content = tf.io.read_file(filename)
    if filename.endswith('.tif') or filename.endswith('.tiff'):
        return tfio.experimental.image.decode_tiff(file_content)
    elif filename.endswith('.png'):
        return tf.image.decode_png(file_content)
    elif filename.endswith('.jpg') or filename.endswith('.jpeg'):
        return tf.image.decode_jpeg(file_content)
    elif filename.endswith('.gif'):
        return tf.image.decode_gif(file_content)
    elif filename.endswith('.bmp'):
        return tf.image.decode_bmp(file_content)
    else:
        raise ValueError('Unsupported image format: {}'.format(filename))


def _parse_function_scene(file_image, label):
    return _parse_image(file_image), label


def _parse_function_object(file_image, targets):
    img = _parse_image(file_image)
    img_height = img.shape[0]
    img_width = img.shape[1]
    for target in targets:
        target[0] = target[0] / img_width
        target[1] = target[1] / img_height
        target[2] = target[2] / img_width
        target[3] = target[3] / img_height
    return img, targets


def _parse_function_segmentation(file_image, file_label_image, color_to_index):
    img = _parse_image(file_image)
    label = cv2.imread(file_label_image, cv2.COLOR_BGR2RGB)
    index_label = utils.label_to_index(label, color_to_index)
    index_label = np.asarray(index_label, dtype=np.uint8)
    index_label = tf.convert_to_tensor(index_label)
    return img, index_label


class TensorflowEOImageSceneTD:
    """
   TensorFlow Dataset for EO image scene classification training dataset
   """

    def __init__(self, td_list, class_map, resize=28):
        self.td_list = td_list
        self.class_map = class_map

    def __len__(self):
        return len(self.td_list)


    def create_dataset(self):
        """
        Create a tensorflow dataset
        """
        img_list = []
        label_list = []
        for td in self.td_list:
            img_list.append(td.data_url)
            label_list.append(self.class_map[td.labels[0].label_class])
        tf_img_list = tf.constant(img_list)
        tf_label_list = tf.constant(label_list)
        dataset = tf.data.Dataset.from_tensor_slices((tf_img_list, tf_label_list))
        dataset = dataset.map(_parse_function_scene)
        return dataset


class TensorflowEOImageObjectTD:
    """
   TensorFlow Dataset for EO image object detection training dataset
   """

    def __init__(self, td_list, class_map):
        self.td_list = td_list
        self.class_map = class_map

    def __len__(self):
        return len(self.td_list)

    def create_dataset(self):
        """
        Create a tensorflow dataset
        """
        img_list = []
        targets_list = []
        for td in self.td_list:
            img_list.append(td.data_url)
            target_list = []
            for label in self.td_list.labels:
                class_value_ = self.class_map[label.label_class]
                target_ = utils.get_bounding_box(label.object)
                target_.append(class_value_)
                target_list.append(target_)
        tf_img_list = tf.constant(img_list)
        tf_targets_list = tf.constant(targets_list)
        dataset = tf.data.Dataset.from_tensor_slices((tf_img_list, tf_targets_list))
        dataset = dataset.map(_parse_function_object)
        return dataset


class TensorflowEOImageSegmentationTD:
    """
   TensorFlow Dataset for EO image semantic segmentation training dataset
   """

    def __init__(self, td_list, class_map):
        self.td_list = td_list
        self.color_to_index = utils.class_to_index(class_map)

    def __len__(self):
        return len(self.td_list)

    def create_dataset(self):
        """
        Create a tensorflow dataset
        """
        img_list = []
        label_img_list = []
        for td in self.td_list:
            img_list.append(td.data_url)
            label_img_list.append(td.labels[0].image_url)
        tf_img_list = tf.constant(img_list)
        tf_label_img_list = tf.constant(label_img_list)
        tf_color_to_index = tf.constant(self.color_to_index)
        dataset = tf.data.Dataset.from_tensor_slices((tf_img_list, tf_label_img_list, tf_color_to_index))
        dataset = dataset.map(_parse_function_segmentation)
        return dataset


from datalibrary.s3Client import minio_client as client
from io import BytesIO
from PIL import Image
import os


def _read_image(root, sample_url):

    file_path = download_scene_data((sample_url, root))
    img = utils.image_open(file_path)

    return img


def _read_image_target(root, sample_url):
    pass


def _process_image(img):
    # img = img.resize((224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img


class TensorSceneClassificationDataPipe:
    def __init__(self, tdml, root, class_map):

        self.class_map = class_map
        self.root = root

        self.td_list = tdml.data
        self.tf_imgs, self.tf_labels = self._load_data()

    def _load_data(self):
        imgs = [item.data_url for item in self.td_list]
        labels = [self.class_map[item.labels[0].label_class] for item in self.td_list]
        return imgs, labels

    def _process_scene_data(self, sample_url, label):

        img = _read_image(self.root, sample_url)
        img = _process_image(img)

        return img, label

    def __len__(self):
        return len(self.td_list)

    def __getitem__(self, idx):
        sample_url = self.tf_imgs[idx]
        label = self.tf_labels[idx]
        img, label = self._process_scene_data(sample_url, label)
        return img, label

    def generator(self):
        for item in zip(self.tf_imgs, self.tf_labels):
            file_path = download_scene_data((item[0], self.root))
            img = utils.image_open(file_path)
            label = item[1]
            yield img, label

    def as_dataset(self, batch_size=32, shuffle=True):

        dataset = tf.data.Dataset.from_generator(self.generator, output_types=(tf.float32, tf.int32))
        # dataset = dataset.map()
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset


class TensorObjectDetectionDataPipe:
    def __init__(self, tdml, root, class_map, crop):

        self.td_list = tdml.data

        self.root = root
        self.class_map = class_map
        self.crop = crop
        self.tf_imgs, self.tf_labels = self._load_data()

    def _load_data(self):
        img_list = []
        targets_list = []
        for td in self.td_list:
            img_list.append(td.data_url)

            targets_list.append(td.labels)

        return img_list, targets_list

    def __len__(self):
        return len(self.td_list)

    def _process_object_data(self, sample_url, targets):
        img = _read_image(self.root, sample_url)
        img_height = img.shape[0]
        img_width = img.shape[1]
        for target in targets:
            target[0] = target[0] / img_width
            target[1] = target[1] / img_height
            target[2] = target[2] / img_width
            target[3] = target[3] / img_height
        return img, targets

    def generator(self):
        for sample_url, labels in zip(self.tf_imgs, self.tf_labels):
            bucket_name, object_name = sample_url.split("/", 1)

            file_path = utils.generate_local_file_path(self.root, sample_url)

            if not os.path.exists(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            if not os.path.exists(file_path):
                client.fget_object(bucket_name, object_name, file_path)

            img = image_open(file_path)

            img_height, img_width, channel = img.shape
            # single band check
            img = utils.channel_processing(img)

            if self.crop is None:
                # 下载文件
                # transform annotations
                targets = utils.transform_annotation(labels, self.class_map, img_width, img_height)
                yield img, targets
            else:

                crop_object = CropWithTargetImage(*self.crop)  # 补充参数
                crop_paths, targets = crop_object(img, labels, os.path.dirname(file_path),
                                         sample_url.split("/")[-1])

                for index, crop_path in enumerate(crop_paths):
                    img = image_open(crop_path)
                    num_targets = []
                    for target in targets[index]:
                        json_object = {"bbox": target["bbox"], "type": "Feature"}
                        labels = [ObjectLabel(object=json_object,
                                              label_class=target["class"],
                                              bbox_type=target["bboxType"],
                                              is_negative=target["isNegative"],
                                              is_difficultly_detectable=target["isDiffDetectable"])]
                        num_target = utils.get_object_label_data_(labels[0], self.class_map, img_width, img_height)
                        num_targets.append(num_target)
                    num_targets = np.asarray(num_targets)

                    yield img, num_targets

    def as_dataset(self, batch_size=32, shuffle=True):
        # dataset = tf.data.Dataset.from_tensor_slices(self.tf_imgs, self.tf_labels)
        # if shuffle:
        #     dataset = dataset.shuffle(buffer_size=len(self.tf_imgs))
        # # dataset = dataset.map(
        #     # lambda x, y: (tf.py_function(func=_read_image, inp=[x], Tout=tf.uint8), y),
        #     # num_parallel_calls=tf.data.AUTOTUNE)
        # dataset = dataset.map(self._process_object_data, num_parallel_calls=tf.data.AUTOTUNE)
        # dataset = dataset.batch(batch_size)
        # dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        # return dataset
        ## 方案二
        dataset = tf.data.Dataset.from_generator(self.generator, output_types=(tf.float32, tf.float32),
                                                 output_shapes=((None, None, 3), (None, 5)))
        # dataset = dataset.map()
        # if shuffle:
        #     dataset = dataset.shuffle(buffer_size=len(self.tf_imgs))
        # dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset


class TensorSemanticSegmentationDataPipe:
    def __init__(self, tdml, root, training_type, classes, crop):
        self.classes = classes
        self.crop = crop

        self.root = root

        self.td_list = tdml.data
        self.tf_imgs, self.tf_labels = self._load_data()

    def _load_data(self):
        img_list = []
        label_img_list = []
        for td in self.td_list:
            img_list.append(td.data_url)
            label_img_list.append(td.labels[0].image_url)
        tf_img_list = tf.constant(img_list)
        tf_label_img_list = tf.constant(label_img_list)

        return tf_img_list, tf_label_img_list

    def __len__(self):
        return len(self.td_list)

    def _process_segmentation_data(self, sample_url, label_url):
        img = _read_image(sample_url, self.root)
        label = _read_image(label_url, self.root)

        index_label = np.asarray(label, dtype=np.uint8)
        index_label = tf.convert_to_tensor(index_label)

        return img, index_label

    def generator(self):
        for sample_url, label_url in zip(self.tf_imgs, self.tf_labels):
            bucket_name, object_name = sample_url.split("/", 1)
            _, label_object_name = label_url.split("/", 1)

            image_path = utils.generate_local_file_path(self.root, sample_url)
            label_path = utils.generate_local_file_path(self.root, label_url)
            try:
                if not os.path.exists(os.path.dirname(image_path)):
                    os.makedirs(os.path.dirname(image_path))
                if not os.path.exists(os.path.dirname(label_path)):
                    os.makedirs(os.path.dirname(label_path))

            except OSError as error:
                print(error)

            if not os.path.exists(image_path):
                client.fget_object(bucket_name, object_name, image_path)
            if not os.path.exists(label_path):
                client.fget_object(bucket_name, object_name, label_path)
            #
            img = image_open(image_path)
            #
            label = image_open(label_path)
            #
            label = utils.regenerate_png_label_(label, self.classes)

            if self.crop is None:

                yield img, label
            else:

                crop_object = CropWithImage(*self.crop)  # 补充参数
                image_crop_paths = crop_object(img, os.path.dirname(image_path), sample_url.split("/")[-1])
                label_crop_paths = crop_object(label, os.path.dirname(label_path), label_url.split("/")[-1])

                for i in range(len(image_crop_paths)):
                    img = image_open(image_crop_paths[i])
                    label = image_open(label_crop_paths[i])
                    yield img, label

    def as_dataset(self, batch_size=32, shuffle=True):
        # dataset = tf.data.Dataset.from_tensor_slices((self.tf_imgs, self.tf_labels))
        # if shuffle:
        #     dataset = dataset.shuffle(buffer_size=len(self.tf_imgs))
        # dataset = dataset.map(
        #     lambda x, y: (tf.py_function(func=_read_image, inp=[x], Tout=tf.uint8), y),
        #     num_parallel_calls=tf.data.AUTOTUNE)
        # dataset = dataset.map(self._process_segmentation_data, num_parallel_calls=tf.data.AUTOTUNE)
        # dataset = dataset.batch(batch_size)
        # dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        # return dataset
        dataset = tf.data.Dataset.from_generator(self.generator, output_types=(tf.float32, tf.float32))
        # dataset = dataset.map()
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.tf_imgs))
        # dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset
