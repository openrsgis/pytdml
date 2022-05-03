# ------------------------------------------------------------------------------
#
# Project: pytdml
#
# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

import pytdml.utils as utils


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
