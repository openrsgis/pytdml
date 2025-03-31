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
import os
import yaml
import s3fs
from pytdml.io import write_to_json
from pytdml.type import EOTrainingDataset, TrainingDataset


def yaml_to_eo_tdml(yaml_path):
    """
    Transform yaml to tdml
    """
    yaml_file = open(yaml_path, "r", encoding="utf-8")
    yaml_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
    eo_training_dataset = EOTrainingDataset.from_dict(yaml_dict)
    return eo_training_dataset


def yaml_to_tdml(yaml_path):
    """
    Transform yaml to tdml
    """
    yaml_file = open(yaml_path, "r", encoding="utf-8")
    yaml_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
    eo_training_dataset = TrainingDataset.from_dict(yaml_dict)
    return eo_training_dataset


# def load_data_scene_label(image_set):
#     """
#     Load training dataset for scene classification
#     """
#     td_list = []
#     file_format = image_set['format']
#     root_path = image_set['root_path']
#     sub_dirs = []
#     if image_set.__contains__('sub_path') and image_set['sub_path'] is not None:  # if set sub path
#         sub_dirs = image_set['sub_path']
#     else:
#         for root, dirs, files in os.walk(root_path):
#             sub_dirs = dirs
#             break
#     try:
#         for sub_dir in sub_dirs:
#             image_path = os.path.join(root_path, sub_dir)
#             for root, dirs, files in os.walk(image_path):
#                 for file in files:
#                     root = str(root)
#                     file = str(file)
#                     data_url = os.path.join(root, file)
#                     if file_format == os.path.splitext(data_url)[-1]:
#                         td = EOTrainingData(
#                             id=file.split(".")[0],
#                             labels=[SceneLabel(label_class=sub_dir)],
#                             data_url=data_url,
#                         )
#                         td_list.append(td)
#         return td_list
#     except IOError:
#         return IOError("Failed to read scene label")


# def load_data_semantic_segmentation(image_set, label_set):
#     """
#     Load training dataset for semantic segmentation
#     """
#     td_list = []
#     image_dir, label_dir = get_image_label_url(image_set, label_set)
#     index = 0
#     for image_url, label_url in zip(image_dir, label_dir):
#         td = EOTrainingData(
#             id=str(index),
#             labels=[PixelLabel(image_url=label_url, image_format=image_set['format'])],
#             data_url=str(image_url)
#         )
#         index = index + 1
#         td_list.append(td)
#     return td_list


# def load_data_change_detection(image_before_set, image_after_set, label_set, data_time):
#     """
#     Load training dataset for change detection
#     """
#     td_list = []
#     image_before_root_path = image_before_set['root_path']
#     image_after_root_path = image_after_set['root_path']
#     label_root_path = label_set['root_path']
#     image_before_format = image_before_set['format']
#     image_after_format = image_after_set['format']
#     label_format = label_set['format']
#     image_before_dir = []
#     image_after_dir = []
#     label_dir = []
#
#     if image_before_set.__contains__('sub_path') and image_before_set['sub_path'] is not None \
#             and image_after_set.__contains__('sub_path') and image_after_set['sub_path'] is not None \
#             and label_set.__contains__('sub_path') and label_set['sub_path'] is not None:
#         image_before_sub_paths = image_before_set['sub_path']
#         image_after_sub_paths = image_after_set['sub_path']
#         label_sub_paths = label_set['sub_path']
#         for image_before_sub_path in image_before_sub_paths:
#             image_before_dir.extend(
#                 traverse_folder(os.path.join(image_before_root_path, image_before_sub_path), image_before_format))
#         for image_after_sub_path in image_after_sub_paths:
#             image_after_dir.extend(
#                 traverse_folder(os.path.join(image_after_root_path, image_after_sub_path), image_after_format))
#         for label_sub_path in label_sub_paths:
#             label_dir.extend(traverse_folder(os.path.join(label_root_path, label_sub_path), label_format))
#     else:
#         image_before_dir = traverse_folder(image_before_root_path, image_before_format)
#         image_after_dir = traverse_folder(image_after_root_path, image_after_format)
#         label_dir = traverse_folder(label_root_path, label_format)
#
#     index = 0
#     for image_before_url, image_after_url, label_url in zip(image_before_dir, image_after_dir, label_dir):
#         td = EOTrainingData(
#             id=str(index),
#             labels=[PixelLabel(image_url=label_url)],
#             data_url=[image_before_url, image_after_url]
#         )
#         td.date_time = [str(data_time[0]), str(data_time[1])]
#         index = index + 1
#         td_list.append(td)
#     return td_list


# def load_data_object_detection(image_set, label_set):
#     """
#     Load training dataset for object detection
#     """
#     td_list = []
#     label_format = label_set['format']
#     image_format = image_set['format']
#
#     if label_format == '.txt':  # txt format
#         image_dir, label_dir = get_image_label_url(image_set, label_set)
#         separate = label_set['separate']
#         label_column = label_set['column']
#         index = 0
#         for image_url, label_url in zip(image_dir, label_dir):
#             labels = read_txt_label(label_url, label_column, separate)
#             td = EOTrainingData(
#                 id=str(index),
#                 labels=labels,
#                 data_url=image_url,
#             )
#             td.number_of_labels = len(td.labels)
#             index = index + 1
#             td_list.append(td)
#     elif label_format == 'stac' and image_format == 'stac':  # stac format
#         images_stac_path = image_set['root_path'] + "\\collection.json"
#         labels_stac_path = label_set['root_path'] + "\\collection.json"
#         images_stac_json = json.load(open(images_stac_path, 'r'))
#         labels_stac_json = json.load(open(labels_stac_path, 'r'))
#         num = len(images_stac_json['links'])
#         for i in range(num):
#             image_stac_path = os.path.join(image_set['root_path'], images_stac_json['links'][i]['href'])
#             label_stac_path = os.path.join(label_set['root_path'], labels_stac_json['links'][i]['href'])
#             image_stac_json = json.load(open(image_stac_path, 'r'))
#             label_stac_json = json.load(open(label_stac_path, 'r'))
#             label_json_path = os.path.join(os.path.dirname(label_stac_path),
#                                            label_stac_json['assets']['labels']['href'])
#             labels = read_geojson_label(label_json_path)
#             td = EOTrainingData(
#                 id=image_stac_json['id'],
#                 extent=image_stac_json['bbox'],
#                 date_time=image_stac_json['properties']['datetime'],
#                 data_url=os.path.join(os.path.dirname(image_stac_path),
#                                       list(image_stac_json['assets'].values())[0]['href']),
#                 labels=labels,
#             )
#             td.number_of_labels = len(td.labels)
#             td_list.append(td)
#         return td_list


# def read_txt_label(csv_path, column_name, separate):
#     """
#     Read the label file in text format in object detection
#     """
#     df = pd.read_csv(csv_path, sep=separate, header=None, names=column_name, engine='python')
#     labels = []
#     is_have_nan = df.isnull().any(axis=1)  # if null
#     for index, row in df.iterrows():
#         if not is_have_nan[index]:
#             polygon = Polygon(
#                 [(float(row['x1']), float(row['y1'])),
#                  (float(row['x2']), float(row['y2'])),
#                  (float(row['x3']), float(row['y3'])),
#                  (float(row['x4']), float(row['y4']))]
#             )
#             label = ObjectLabel(
#                 object=Feature(geometry=polygon),
#                 bbox_type='Horizontal BBox',
#                 label_class=row['class'],
#                 is_difficultly_detectable=row['isDiffDetectable']
#             )
#             labels.append(label)
#     return labels


# def read_geojson_label(geojson_path):
#     """
#     Read the label file in geojson format in object detection
#     """
#     fc = json.load(open(geojson_path))
#     labels = []
#     for f in fc['features']:
#         labels.append(ObjectLabel(
#             object=f
#         ))
#     return labels


def traverse_folder(file_path, file_format):
    """
    Traverse target folder
    """
    try:
        file_dir = []
        for root, dirs, files in os.walk(file_path):
            for f in sorted(files):
                if file_format == os.path.splitext(f)[-1]:
                    file_dir.append(os.path.join(root, f))
        return file_dir
    except IOError:
        return IOError("Failed to load dataset")


def traverse_s3(object_path, file_format):
    """
    Traverse s3 object path
    """
    try:
        s3_object = []
        s3 = s3fs.S3FileSystem(anon=True)
        split = object_path.split("/")
        bucket = split[2]
        for root, dirs, files in s3.walk(bucket):
            for f in files:
                if (
                    (split[3] in root)
                    and (split[4] in root)
                    and (split[5] in root)
                    and (file_format == os.path.splitext(f)[-1])
                ):
                    s3_object.append(os.path.join("s3://" + root, f))
        return s3_object
    except IOError:
        return IOError("Failed to load dataset")


def get_image_label_url(image_set, label_set):
    """
    Get image url and label url from image set and label set
    """
    image_root_path = image_set["root_path"]
    label_root_path = label_set["root_path"]
    image_format = image_set["format"]
    label_format = label_set["format"]
    image_dir = []
    label_dir = []
    # if set sub path
    if (
        image_set.__contains__("sub_path")
        and image_set["sub_path"] is not None
        and label_set.__contains__("sub_path")
        and label_set["sub_path"] is not None
    ):
        image_sub_paths = image_set["sub_path"]
        label_sub_paths = label_set["sub_path"]
        for image_sub_path in image_sub_paths:
            image_dir.extend(
                traverse_folder(
                    os.path.join(image_root_path, image_sub_path), image_format
                )
            )
        for label_sub_path in label_sub_paths:
            label_dir.extend(
                traverse_folder(
                    os.path.join(label_root_path, label_sub_path), label_format
                )
            )
    elif "s3" in image_root_path:  # s3 buckets being used
        image_dir = traverse_s3(image_root_path, image_format)
        label_dir = traverse_s3(label_root_path, label_format)
    else:
        image_dir = traverse_folder(image_root_path, image_format)
        label_dir = traverse_folder(label_root_path, label_format)
    return image_dir, label_dir


def main():
    parser = argparse.ArgumentParser(
        description="Encode a training dataset to TrainingDML-AI JSON format based on "
        "YAML configuration file"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="YAML configuration file path"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output TrainingDML-AI JSON file path"
    )
    parser.add_argument(
        "--format",
        type=str,
        required=True,
        choices=["EO-TDML", "TDML"],
        help="Specify the output format",
    )

    args = parser.parse_args()
    yaml_path = args.config
    json_path = args.output
    if args.format == "EO-TDML":
        training_datasets = yaml_to_eo_tdml(yaml_path)
    else:
        training_datasets = yaml_to_tdml(yaml_path)
    if training_datasets:
        write_to_json(training_datasets, json_path)


if __name__ == "__main__":
    main()
