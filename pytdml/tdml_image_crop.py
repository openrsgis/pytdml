# ------------------------------------------------------------------------------
#
# Project: pytdml
#
# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
import json
import os
import sys

import cv2

from pytdml.io import read_from_json
from pytdml.type import EOTrainingDataset, EOTrainingData, PixelLabel
from pytdml.utils import remove_empty


def td_image_crop(td: EOTrainingDataset, save_tdml_path: str, save_crop_dir: str, sub_size: int):
    td_dict = td.to_dict()
    td_list = td_dict['data']
    new_td_list = []
    index = 0
    for d in td_list:
        label_type = d['labels'][0]['type']
        if label_type == 'PixelLabel':
            image_url = d['dataUrl']
            label_url = d['labels'][0]['imageUrl']
            image_dir = os.path.join(save_crop_dir, "images")
            label_dir = os.path.join(save_crop_dir, "labels")
            if not os.path.isdir(image_dir):
                os.makedirs(image_dir)
            if not os.path.isdir(label_dir):
                os.makedirs(label_dir)
            crop_image_list = image_crop(image_url, image_dir, sub_size)
            crop_label_list = image_crop(label_url, label_dir, sub_size)
            for crop_image_url, crop_label_url in zip(crop_image_list, crop_label_list):
                new_d = EOTrainingData(
                    id=str(index),
                    labels=[PixelLabel(image_url=crop_label_url)],
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


def main(argv=None):
    if len(argv) < 4:
        print("Please set parameters "
              "<*input tdml json file path> "
              "<*save path of new tdml json file> "
              "<*save dir of result images and labels> "
              "<*crop size>")
        return
    tdml_path = argv[1]
    save_tdml_path = argv[2]
    save_crop_dir = argv[3]
    sub_size = int(argv[4])
    training_datasets = read_from_json(tdml_path)
    if training_datasets:
        td_image_crop(training_datasets, save_tdml_path, save_crop_dir, sub_size)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
