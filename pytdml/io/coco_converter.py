import json
import os
import re
from datetime import datetime
from geojson import Feature

from pytdml.type import AI_EOTrainingData, EOTrainingDataset, AI_EOTask, AI_ObjectLabel, AI_PixelLabel, AI_SceneLabel
from pytdml.type.basic_types import NamedValue


def categorize_string(s):
    """
    检查字符串 s 是否包含特定单词（train, test, valid），并返回相应类别。

    参数:
        s (str): 要检查的字符串。

    返回:
        str: 匹配的类别。
    """
    if re.search(r'train', s, re.IGNORECASE):
        return 'train'
    elif re.search(r'test', s, re.IGNORECASE):
        return 'test'
    elif re.search(r'valid', s, re.IGNORECASE):
        return 'validation'
    else:
        return 'unknown'  # If none of the above matches, then 'unknown' or some other default value is returned.


def distinguish_dataset_type(coco_dataset):
    if not(coco_dataset.keys().__contains__("annotations")):
        return "Unknown"

    anns = coco_dataset['annotations']
    if anns[0].keys().__contains__("keypoints"):
        return "Keypoint Detection"
    elif anns[0].keys().__contains__("segments_info"):
        return "Panoptic Segmentation"
    elif anns[0].keys().__contains__("dp_masks"):
        return "Dense Segmentation"
    elif anns[0].keys().__contains__("caption"):
        return "Image Captioning"
    else:
        if 1 <= anns[0]["category_id"] <= 91:
            return "Object Detection"
        else:
            return "Stuff Segmentation"


def update_keypoint_counts(keypoint_counts, keypoints):
    classes = list(keypoint_counts.keys())
    type_class = []
    for i in range(0, len(keypoints) - 1, 3):
        if keypoints[i] != 0:
            keypoint_counts[classes[int(i / 3)]] += 1
            type_class.append(classes[int(i / 3)])
        else:
            continue
    return keypoint_counts, type_class


def parse_date(input_date):
    formats = [
        "%Y-%m-%dT%H:%M:%S",  # date-time
        "%Y-%m-%d",  # date
        "%H:%M:%S",  # time
        "%Y/%m/%d",
        "%Y-%m-%d %H:%M:%S.%f",  # date-time with milliseconds
        "%Y-%m-%d %H:%M:%S"
    ]

    for date_format in formats:
        try:
            parsed_date = datetime.strptime(input_date, date_format)
            return parsed_date.strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue

    return "Invalid date format"


def convert_coco_to_tdml(cocofile):
    """
    Reads data from a COCO-formatted JSON file and converts it to a TDML object.

    params:
        coco_dataset_path (str): JSON file in COCO format

    return:
        EOTrainingDataset
    """

    coco_dataset = json.load(cocofile)

    try:
        info = coco_dataset["info"]
        images = coco_dataset["images"]
        licenses = coco_dataset["licenses"]
    except KeyError as e:
        raise ValueError("The format of the data is incorrect, missing the key {}".format(e))
    except Exception as e:
        raise e

    dataset_type = distinguish_dataset_type(coco_dataset)
    if dataset_type == "Object Detection" or dataset_type == "Stuff Segmentation":
        annotations = coco_dataset.get("annotations")
        annotations_grouped_by_image = {}
        for annotation in annotations:
            image_id = annotation['image_id']
            if image_id not in annotations_grouped_by_image:
                annotations_grouped_by_image[image_id] = []
            annotations_grouped_by_image[image_id].append(annotation)

        categories = coco_dataset.get("categories")
        categories_by_id = {category["id"]: category for category in categories}
        dataset_id = info.get("description")
        dataset_name = dataset_id
        dataset_description = dataset_id
        # license
        names = [license_elem.get("name") for license_elem in licenses]
        dataset_licenses = ', '.join(names)
        # task
        dataset_task = AI_EOTask(
                id=str(dataset_id) + "_task",
                type='AI_EOTask',
                task_type=dataset_type,
                dataset_id=str(dataset_id),
                description="The COCO {} Task.".format(dataset_type)
        )
        # data
        td_list = []
        for image in images:
            labels_list = annotations_grouped_by_image.get(image["id"])
            object_labels = []
            formatted_date_time = parse_date(image["date_captured"])
            if labels_list:
                for i, label_element in enumerate(labels_list):
                    points = label_element.get("bbox")
                    coord = [
                        [points[0], points[1]],
                        [points[0] + points[2], points[1]],
                        [points[0] + points[2], points[1] + points[3]],
                        [points[0], points[1] + points[2]]
                    ]
                    labels = AI_ObjectLabel(
                        type="AI_ObjectLabel",
                        is_negative=False,
                        confidence =1.0,
                        object=Feature(
                            id="feature " + str(i),
                            geometry={
                                "type": "Polygon",
                                "coordinates": coord
                            }
                        ),
                        label_class=categories_by_id[label_element["category_id"]].get("name"),
                        bbox_type="Horizontal BBox")
                    object_labels.append(labels)

            training_type = categorize_string(os.path.basename(os.path.dirname(image["coco_url"])))
            numbers_of_labels = len(object_labels)

            td = AI_EOTrainingData(
                type="AI_EOTrainingData",
                id=str(image["id"]),
                labels=object_labels,
                data_URL=[image.get("coco_url")],
                dataset_id=dataset_id,
                number_of_labels=numbers_of_labels,
                training_type=training_type,
                date_time=[formatted_date_time]
                )
            td_list.append(td)

        amount_of_trainingdata = len(images)

        # classes
        category_counts = {}
        for annotation in annotations:
            category_id = annotation["category_id"]
            category_name = categories_by_id[category_id].get("name")
            if category_name in category_counts.keys():
                category_counts[category_name] += 1
            else:
                category_counts[category_name] = 1

        classes = [
            NamedValue(
                key=category["name"],
                value=category_counts.get(category["name"])
            ) for category in categories
        ]

        created_time = parse_date(info.get("date_created"))
        updated_time = created_time
        number_of_classes = len(classes)
        dataset_version = info.get("version")
        providers = [info.get("contributor")]

        dataset = EOTrainingDataset(
            id=str(dataset_id),
            name=dataset_name,
            description=dataset_description,
            license=dataset_licenses,
            tasks=[dataset_task],
            data=td_list,
            type="AI_EOTrainingDataset",
            amount_of_training_data=amount_of_trainingdata,
            classes=classes,
            created_time=created_time,
            number_of_classes=number_of_classes,
            providers=providers,
            updated_time=updated_time,
            version=dataset_version
        )

        return dataset

    elif dataset_type == "Keypoint Detection":
        annotations = coco_dataset.get("annotations")
        annotations_grouped_by_image = {}
        for annotation in annotations:
            image_id = annotation['image_id']
            if image_id not in annotations_grouped_by_image:
                annotations_grouped_by_image[image_id] = []
            annotations_grouped_by_image[image_id].append(annotation)
        categories = coco_dataset.get("categories")
        dataset_id = info.get("description")
        dataset_name = dataset_id
        dataset_description = dataset_id
        # license
        names = [license_elem.get("name") for license_elem in licenses]
        dataset_licenses = ', '.join(names)
        # task
        dataset_task = AI_EOTask(
                id=str(dataset_id) + "_task",
                type='AI_EOTask',
                task_type=dataset_type,
                dataset_id=str(dataset_id),
                description="The COCO {} Task.".format(dataset_type)
        )
        # data
        td_list = []
        # classes statistics
        keypoints = categories[0]["keypoints"]
        keypoint_counts = {keypoint: 0 for keypoint in keypoints}

        for image in images:
            labels_list = annotations_grouped_by_image.get(image["id"])
            object_labels = []
            formatted_date_time = parse_date(image["date_captured"])
            if labels_list:
                for i, label_element in enumerate(labels_list):
                    points = label_element.get("bbox")
                    coord = [
                        [points[0], points[1]],
                        [points[0] + points[2], points[1]],
                        [points[0] + points[2], points[1] + points[3]],
                        [points[0], points[1] + points[2]]
                    ]
                    keypoint_counts, label_class = update_keypoint_counts(keypoint_counts, label_element["keypoints"])
                    labels = AI_ObjectLabel(
                        type="AI_ObjectLabel",
                        is_negative=False,
                        confidence =1.0,
                        object=Feature(
                            id="feature " + str(i),
                            geometry={
                                "type": "Polygon",
                                "coordinates": coord
                            }
                        ),
                        label_class=", ".join(label_class),
                        bbox_type="Horizontal BBox")
                    object_labels.append(labels)

            training_type = categorize_string(os.path.basename(os.path.dirname(image["coco_url"])))
            numbers_of_labels = len(object_labels)

            td = AI_EOTrainingData(
                type="AI_EOTrainingData",
                id=str(image["id"]),
                labels=object_labels,
                data_URL=[image.get("coco_url")],
                dataset_id=dataset_id,
                number_of_labels=numbers_of_labels,
                training_type=training_type,
                date_time=[formatted_date_time]
                )
            td_list.append(td)

        amount_of_trainingdata = len(images)
        created_time = parse_date(info.get("date_created"))
        updated_time = created_time
        classes = [
            NamedValue(
                key=keypoint,
                value=keypoint_counts.get(keypoint)
            ) for keypoint in keypoints
        ]
        number_of_classes = len(classes)

        dataset_version = info.get("version")
        providers = [info.get("contributor")]

        dataset = EOTrainingDataset(
            id=str(dataset_id),
            name=dataset_name,
            description=dataset_description,
            license=dataset_licenses,
            tasks=[dataset_task],
            data=td_list,
            type="AI_EOTrainingDataset",
            amount_of_training_data=amount_of_trainingdata,
            classes=classes,
            created_time=created_time,
            number_of_classes=number_of_classes,
            providers=providers,
            updated_time=updated_time,
            version=dataset_version
        )

        return dataset

    elif dataset_type == "Panoptic Segmentation":
        annotations = coco_dataset.get("annotations")
        annotations_by_image_id = {annotation["image_id"]: annotation for annotation in annotations}
        categories = coco_dataset.get("categories")
        categories_by_id = {category["id"]: category for category in categories}
        dataset_id = info.get("description")
        dataset_name = dataset_id
        dataset_description = dataset_id
        # license
        names = [license_elem.get("name") for license_elem in licenses]
        dataset_licenses = ', '.join(names)
        # task
        dataset_task = AI_EOTask(
                id=str(dataset_id) + "_task",
                type='AI_EOTask',
                task_type=dataset_type,
                dataset_id=str(dataset_id),
                description="The COCO {} Task.".format(dataset_type)
        )
        # data
        td_list = []
        for image in images:
            object_labels = []
            formatted_date_time = parse_date(image["date_captured"])
            labels_list = annotations_by_image_id[image["id"]]["segments_info"]
            if labels_list:
                for i, label_element in enumerate(labels_list):
                    points = label_element.get("bbox")
                    coord = [
                        [points[0], points[1]],
                        [points[0] + points[2], points[1]],
                        [points[0] + points[2], points[1] + points[3]],
                        [points[0], points[1] + points[2]]
                    ]
                    labels = AI_ObjectLabel(
                        type="AI_ObjectLabel",
                        is_negative=False,
                        confidence =1.0,
                        object=Feature(
                            id="feature " + str(i),
                            geometry={
                                "type": "Polygon",
                                "coordinates": coord
                            }
                        ),
                        label_class=categories_by_id[label_element["category_id"]].get("name"),
                        bbox_type="Horizontal BBox")
                    object_labels.append(labels)

            training_type = categorize_string(os.path.basename(os.path.dirname(image["coco_url"])))
            numbers_of_labels = len(object_labels)

            td = AI_EOTrainingData(
                type="AI_EOTrainingData",
                id=str(image["id"]),
                labels=object_labels,
                data_URL=[image.get("coco_url")],
                dataset_id=dataset_id,
                number_of_labels=numbers_of_labels,
                training_type=training_type,
                date_time=[formatted_date_time]
                )
            td_list.append(td)

        amount_of_trainingdata = len(images)

        # classes
        category_counts = {}
        for annotation in annotations:
            for segments_info in annotation["segments_info"]:
                category_id = segments_info["category_id"]
                category_name = categories_by_id[category_id].get("name")
                if category_name in category_counts.keys():
                    category_counts[category_name] += 1
                else:
                    category_counts[category_name] = 1

        classes = [
            NamedValue(
                key=category["name"],
                value=category_counts.get(category["name"])
            ) for category in categories
        ]

        created_time = parse_date(info.get("date_created"))
        number_of_classes = len(classes)
        dataset_version = info.get("version")
        providers = [info.get("contributor")]

        dataset = EOTrainingDataset(
            id=str(dataset_id),
            name=dataset_name,
            description=dataset_description,
            license=dataset_licenses,
            tasks=[dataset_task],
            data=td_list,
            type="AI_EOTrainingDataset",
            amount_of_training_data=amount_of_trainingdata,
            classes=classes,
            created_time=created_time,
            number_of_classes=number_of_classes,
            providers=providers,
            updated_time=created_time,
            version=dataset_version
        )

        return dataset

    elif dataset_type == "Image Captioning":
        annotations = coco_dataset.get("annotations")
        annotations_grouped_by_image = {}
        for annotation in annotations:
            image_id = annotation['image_id']
            if image_id not in annotations_grouped_by_image:
                annotations_grouped_by_image[image_id] = []
            annotations_grouped_by_image[image_id].append(annotation)
        dataset_id = info.get("description")
        dataset_name = dataset_id
        dataset_description = dataset_id
        # license
        names = [license_elem.get("name") for license_elem in licenses]
        dataset_licenses = ', '.join(names)
        # task
        dataset_task = AI_EOTask(
                id=str(dataset_id) + "_task",
                type='AI_EOTask',
                task_type=dataset_type,
                dataset_id=str(dataset_id),
                description="The COCO {} Task.".format(dataset_type)
        )
        # data
        td_list = []
        for image in images:
            object_labels = []
            formatted_date_time = parse_date(image["date_captured"])
            labels_list = annotations_grouped_by_image.get(image["id"])
            if labels_list:
                for label_element in labels_list:
                    labels = AI_SceneLabel(
                        type="AI_SceneLabel",
                        label_class=label_element.get("caption"),
                        is_negative=False,
                        confidence =1.0
                    )
                    object_labels.append(labels)

            training_type = categorize_string(os.path.basename(os.path.dirname(image["coco_url"])))
            numbers_of_labels = len(object_labels)

            td = AI_EOTrainingData(
                type="AI_EOTrainingData",
                id=str(image["id"]),
                labels=object_labels,
                data_URL=[image.get("coco_url")],
                dataset_id=dataset_id,
                number_of_labels=numbers_of_labels,
                training_type=training_type,
                date_time=[formatted_date_time]
                )
            td_list.append(td)

        amount_of_trainingdata = len(images)
        created_time = parse_date(info.get("date_created"))
        dataset_version = info.get("version")
        providers = [info.get("contributor")]

        dataset = EOTrainingDataset(
            id=str(dataset_id),
            name=dataset_name,
            description=dataset_description,
            license=dataset_licenses,
            tasks=[dataset_task],
            data=td_list,
            type="AI_EOTrainingDataset",
            amount_of_training_data=amount_of_trainingdata,
            created_time=created_time,
            providers=providers,
            updated_time=created_time,
            version=dataset_version
        )

        return dataset

    else:
        annotations = coco_dataset.get("annotations")
        annotations_grouped_by_image = {}
        for annotation in annotations:
            image_id = annotation['image_id']
            if image_id not in annotations_grouped_by_image:
                annotations_grouped_by_image[image_id] = []
            annotations_grouped_by_image[image_id].append(annotation)

        categories = coco_dataset.get("categories")
        categories_by_id = {category["id"]: category for category in categories}
        dataset_id = info.get("description")
        dataset_name = dataset_id
        dataset_description = dataset_id
        # license
        names = [license_elem.get("name") for license_elem in licenses]
        dataset_licenses = ', '.join(names)
        # task
        dataset_task = AI_EOTask(
            id=str(dataset_id) + "_task",
            type='AI_EOTask',
            task_type=dataset_type,
            dataset_id=str(dataset_id),
            description="The COCO {} Task.".format(dataset_type)
        )
        # data
        td_list = []
        for image in images:
            labels_list = annotations_grouped_by_image.get(image["id"])
            object_labels = []
            formatted_date_time = parse_date(image["date_captured"])
            if labels_list:
                for i, label_element in enumerate(labels_list):
                    points = label_element.get("bbox")
                    coord = [
                        [points[0], points[1]],
                        [points[0] + points[2], points[1]],
                        [points[0] + points[2], points[1] + points[3]],
                        [points[0], points[1] + points[2]]
                    ]
                    labels = AI_ObjectLabel(
                        type="AI_ObjectLabel",
                        is_negative=False,
                        confidence=1.0,
                        object=Feature(
                            id="feature " + str(i),
                            geometry={
                                "type": "Polygon",
                                "coordinates": coord
                            }
                        ),
                        label_class=categories_by_id[label_element["category_id"]].get("name"),
                        bbox_type="Horizontal BBox")
                    object_labels.append(labels)

            training_type = categorize_string(os.path.basename(os.path.dirname(image["coco_url"])))
            numbers_of_labels = len(object_labels)

            td = AI_EOTrainingData(
                type="AI_EOTrainingData",
                id=str(image["id"]),
                labels=object_labels,
                data_URL=[image.get("coco_url")],
                dataset_id=dataset_id,
                number_of_labels=numbers_of_labels,
                training_type=training_type,
                date_time=[formatted_date_time]
            )
            td_list.append(td)

        amount_of_trainingdata = len(images)

        # classes
        category_counts = {}
        for annotation in annotations:
            category_id = annotation["category_id"]
            category_name = categories_by_id[category_id].get("name")
            if category_name in category_counts.keys():
                category_counts[category_name] += 1
            else:
                category_counts[category_name] = 1

        classes = [
            NamedValue(
                key=category["name"],
                value=category_counts.get(category["name"])
            ) for category in categories
        ]

        created_time = parse_date(info.get("date_created"))
        updated_time = created_time
        number_of_classes = len(classes)
        dataset_version = info.get("version")
        providers = [info.get("contributor")]

        dataset = EOTrainingDataset(
            id=str(dataset_id),
            name=dataset_name,
            description=dataset_description,
            license=dataset_licenses,
            tasks=[dataset_task],
            data=td_list,
            type="AI_EOTrainingDataset",
            amount_of_training_data=amount_of_trainingdata,
            classes=classes,
            created_time=created_time,
            number_of_classes=number_of_classes,
            providers=providers,
            updated_time=updated_time,
            version=dataset_version
        )

        return dataset


