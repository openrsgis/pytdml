import os
import sys

import yaml
import pandas as pd
from geojson import Polygon

from pytdml.io import write_to_json
from pytdml.type import EOTask, EODataSource, EOTrainingDataset, EOTrainingData, SceneLabel, PixelLabel, ObjectLabel


def yaml_to_eo_tdml(yaml_path):
    """
    Transform yaml to tdml
    """
    try:
        yaml_file = open(yaml_path, "r", encoding='utf-8')
        yaml_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
        dataset_type = yaml_dict['dataset_type']
        tasks_list = yaml_dict['tasks']
        data_sources_list = yaml_dict['data_sources']
        td_list = load_data(yaml_dict['data'])

        tasks = []
        for task in tasks_list:
            eo_task = EOTask(
                task_type=task.__contains__('task_type') and task['task_type'] or "",
                description=task.__contains__('description') and task['description'] or ""
            )
            tasks.append(eo_task)

        data_sources = []
        for data_source in data_sources_list:
            eo_data_source = EODataSource(
                id=data_source.__contains__('id') and data_source['id'] or "",
                data_type=data_source.__contains__('data_type') and data_source['data_type'] or "",
                platform=data_source.__contains__('platform') and data_source['platform'] or "",
                sensor=data_source.__contains__('sensor') and data_source['sensor'] or "",
                citation=data_source.__contains__('citation') and data_source['citation'] or "",
                resolution=data_source.__contains__('resolution') and data_source['resolution'] or "",
            )
            data_sources.append(eo_data_source)

        if dataset_type == 'EOTrainingDataset':
            eo_training_dataset = EOTrainingDataset(
                id=yaml_dict['id'],
                name=yaml_dict['name'],
                description=yaml_dict['description'],
                tasks=tasks,
                data=td_list,
                version=yaml_dict['version'],
                amount_of_training_data=len(td_list),
                created_time=yaml_dict['created_time'],
                updated_time=yaml_dict['updated_time'],
                providers=yaml_dict["providers"],
                keywords=yaml_dict["keywords"],
                data_sources=data_sources,
                classes=yaml_dict['classes'],
                number_of_classes=len(yaml_dict['classes']),
                bands=yaml_dict['bands'],
                image_size=yaml_dict['image_size']
            )
            return eo_training_dataset
    except KeyError:
        print("Invalid EOTrainingDataset yaml")
        return None


def load_data(data_dict):
    """
    Load training dataset
    """
    task_type = data_dict['task_type']
    label_type = data_dict['label_type']
    data_path = data_dict['data_path']

    if label_type == 'SceneLabel' and task_type == 'SceneClassification':
        for data in data_path:
            if data['type'] == 'image':
                return load_data_scene_label(data)
    elif label_type == 'ObjectLabel' and task_type == 'ObjectDetection':
        image_set = {}
        label_set = {}
        for data in data_path:
            if data['type'] == 'image':
                image_set = data
            elif data['type'] == 'label':
                label_set = data
        return load_data_object_detection(image_set, label_set)
    elif label_type == 'PixelLabel' and task_type == 'SemanticSegmentation':
        image_set = {}
        label_set = {}
        for data in data_path:
            if data['type'] == 'image':
                image_set = data
            elif data['type'] == 'label':
                label_set = data
        return load_data_semantic_segmentation(image_set, label_set)
    elif label_type == 'PixelLabel' and task_type == 'ChangeDetection':
        image_before_set = {}
        image_after_set = {}
        label_set = {}
        data_time = data_dict['data_time']
        for data in data_path:
            if data['type'] == 'image_before':
                image_before_set = data
            elif data['type'] == 'image_after':
                image_after_set = data
            elif data['type'] == 'label':
                label_set = data
        return load_data_change_detection(image_before_set, image_after_set, label_set, data_time)


def load_data_scene_label(image_set):
    """
    Load training dataset for scene classification
    """
    td_list = []
    file_format = image_set['format']
    root_path = image_set['root_path']
    sub_dirs = []
    if image_set.__contains__('sub_path') and image_set['sub_path'] is not None:  # if set sub path
        sub_dirs = image_set['sub_path']
    else:
        for root, dirs, files in os.walk(root_path):
            sub_dirs = dirs
            break
    try:
        for sub_dir in sub_dirs:
            image_path = os.path.join(root_path, sub_dir)
            for root, dirs, files in os.walk(image_path):
                for file in files:
                    data_url = os.path.join(root, file)
                    if file_format == os.path.splitext(data_url)[-1]:
                        td = EOTrainingData(
                            id=file.split(".")[0],
                            labels=[SceneLabel(label_class=sub_dir)],
                            data_url=data_url,
                        )
                        td_list.append(td)
        return td_list
    except IOError:
        return IOError("Failed to read scene label")


def load_data_semantic_segmentation(image_set, label_set):
    """
    Load training dataset for semantic segmentation
    """
    td_list = []
    image_dir, label_dir = get_image_label_url(image_set, label_set)
    index = 0
    for image_url, label_url in zip(image_dir, label_dir):
        td = EOTrainingData(
            id=str(index),
            labels=[PixelLabel(image_url=label_url)],
            data_url=image_url
        )
        index = index + 1
        td_list.append(td)
    return td_list


def load_data_change_detection(image_before_set, image_after_set, label_set, data_time):
    """
    Load training dataset for change detection
    """
    td_list = []
    image_before_root_path = image_before_set['root_path']
    image_after_root_path = image_after_set['root_path']
    label_root_path = label_set['root_path']
    image_before_format = image_before_set['format']
    image_after_format = image_after_set['format']
    label_format = label_set['format']
    image_before_dir = []
    image_after_dir = []
    label_dir = []

    if image_before_set.__contains__('sub_path') and image_before_set['sub_path'] is not None \
            and image_after_set.__contains__('sub_path') and image_after_set['sub_path'] is not None \
            and label_set.__contains__('sub_path') and label_set['sub_path'] is not None:
        image_before_sub_paths = image_before_set['sub_path']
        image_after_sub_paths = image_after_set['sub_path']
        label_sub_paths = label_set['sub_path']
        for image_before_sub_path in image_before_sub_paths:
            image_before_dir.extend(
                traverse_folder(os.path.join(image_before_root_path, image_before_sub_path), image_before_format))
        for image_after_sub_path in image_after_sub_paths:
            image_after_dir.extend(
                traverse_folder(os.path.join(image_after_root_path, image_after_sub_path), image_after_format))
        for label_sub_path in label_sub_paths:
            label_dir.extend(traverse_folder(os.path.join(label_root_path, label_sub_path), label_format))
    else:
        image_before_dir = traverse_folder(image_before_root_path, image_before_format)
        image_after_dir = traverse_folder(image_after_root_path, image_after_format)
        label_dir = traverse_folder(label_root_path, label_format)

    index = 0
    for image_before_url, image_after_url, label_url in zip(image_before_dir, image_after_dir, label_dir):
        td = EOTrainingData(
            id=str(index),
            labels=[PixelLabel(image_url=label_url)],
            data_url=[image_before_url, image_after_url]
        )
        td.date_time = [str(data_time[0]), str(data_time[1])]
        index = index + 1
        td_list.append(td)
    return td_list


def load_data_object_detection(image_set, label_set):
    """
    Load training dataset for object detection
    """
    td_list = []
    label_format = label_set['format']
    label_column = label_set['column']
    image_dir, label_dir = get_image_label_url(image_set, label_set)

    index = 0
    if label_format == '.txt':  # txt format
        separate = label_set['separate']
        for image_url, label_url in zip(image_dir, label_dir):
            labels = read_txt_label(label_url, label_column, separate)
            td = EOTrainingData(
                id=str(index),
                labels=labels,
                data_url=image_url,
            )
            td.number_of_Labels = len(td.labels)
            index = index + 1
            td_list.append(td)
        return td_list


def read_txt_label(csv_path, column_name, separate):
    """
    Read the label file in text format in object detection
    """
    df = pd.read_csv(csv_path, sep=separate, header=None, names=column_name, engine='python')
    labels = []
    is_have_nan = df.isnull().any(axis=1)  # if null
    for index, row in df.iterrows():
        if not is_have_nan[index]:
            polygon = Polygon(
                [(float(row['x1']), float(row['y1'])),
                 (float(row['x2']), float(row['y2'])),
                 (float(row['x3']), float(row['y3'])),
                 (float(row['x4']), float(row['y4']))]
            )
            label = ObjectLabel(
                object=polygon,
                geometry_type='Horizontal BBox',
                label_class=row['class'],
                is_difficultly_detectable=row['isDiffDetectable']
            )
            labels.append(label)
    return labels


def traverse_folder(file_path, file_format):
    """
    Traverse target folder
    """
    try:
        file_dir = []
        for root, dirs, files in os.walk(file_path):
            for f in files:
                if file_format == os.path.splitext(f)[-1]:
                    file_dir.append(os.path.join(root, f))
        return file_dir
    except IOError:
        return IOError("Failed to load dataset")


def get_image_label_url(image_set, label_set):
    """
    Get image url and label url from image set and label set
    """
    image_root_path = image_set['root_path']
    label_root_path = label_set['root_path']
    image_format = image_set['format']
    label_format = label_set['format']
    image_dir = []
    label_dir = []
    # if set sub path
    if image_set.__contains__('sub_path') and image_set['sub_path'] is not None \
            and label_set.__contains__('sub_path') and label_set['sub_path'] is not None:
        image_sub_paths = image_set['sub_path']
        label_sub_paths = label_set['sub_path']
        for image_sub_path in image_sub_paths:
            image_dir.extend(traverse_folder(os.path.join(image_root_path, image_sub_path), image_format))
        for label_sub_path in label_sub_paths:
            label_dir.extend(traverse_folder(os.path.join(label_root_path, label_sub_path), label_format))
    else:
        image_dir = traverse_folder(image_root_path, image_format)
        label_dir = traverse_folder(label_root_path, label_format)
    return image_dir, label_dir


def main(argv=None):
    if len(argv) < 2:
        print("Please set parameters <*input yaml file path> <output json file path>")
        return
    yaml_path = argv[1]
    json_path = argv[2]
    training_datasets = yaml_to_eo_tdml(yaml_path)
    if training_datasets:
        write_to_json(training_datasets, json_path)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
