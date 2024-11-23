import json

from geojson import Feature

from pytdml.io import write_to_json
from pytdml.type import AI_EOTrainingData, EOTrainingDataset, AI_EOTask, AI_ObjectLabel, AI_PixelLabel
import os
import re
import time


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


def convert_coco_to_tdml(coco_dataset_path, output_json_path):
    """
    Reads data from a COCO-formatted JSON file and saves it after converting it to a new JSON document.

    params:
        coco_dataset_path (str): COCO 格式 JSON 文件的路径。
        output_json_path (str): 转换后输出 JSON 文件的路径。

    return:
        None
    """

    # Reads JSON data in COCO format from a given path.
    with open(coco_dataset_path, 'r') as cocofile:
        coco_dataset = json.load(cocofile)

    # Create a dictionary to store all the information in the COCO dataset

    # info = coco_dataset.get("info", {}),
    info = coco_dataset["info"]
    licenses = coco_dataset["licenses"]
    images = coco_dataset["images"]
    annotations = coco_dataset["annotations"]
    categories = coco_dataset["categories"]

    dataset_id = info.get("description")
    dataset_description = dataset_id
    dataset_name = dataset_id
    print("dataset name: " + dataset_name)
    dataset_version = info.get("version")
    amount_of_trainingdata = len(images)
    print("amount_of_trainingdata: " + str(amount_of_trainingdata))
    created_time = info.get("date_created").replace('/', '-')
    updated_time = created_time
    providers = [info.get("contributor")]
    keywords = []
    ## license
    names = [license_ele["name"] for license_ele in licenses]

    dataset_licenses = ', '.join(names)

    # classes
    classes = [category["name"] for category in categories]
    number_of_classes = len(classes)
    print("number of classes: " + str(number_of_classes))

    # Convert to dictionary format with id as key
    categories_by_id = {category["id"]: category for category in categories}

    # Group annotations by image_id
    annotations_grouped_by_image = {}
    for annotation in annotations:
        image_id = annotation['image_id']
        if image_id not in annotations_grouped_by_image:
            annotations_grouped_by_image[image_id] = []
        annotations_grouped_by_image[image_id].append(annotation)

    # data
    td_list = []

    # start of timer
    start_time = time.time()

    for image_json in images:

        labels_list = annotations_grouped_by_image.get(image_json["id"])
        object_labels = []
        if labels_list is not None:
            for i,label_element in enumerate(labels_list):
                points = label_element["bbox"]
                coord = [[points[0], points[1]], [points[0] + points[2], points[1]], [points[0] + points[2],
                                                                                      points[1] + points[3]],
                         [points[0], points[1] + points[2]]]
                labels = AI_ObjectLabel(is_negative=False, type="AI_ObjectLabel", confidence=1.0, object=Feature(
                    id="feature " + str(i), geometry={
                        "type": "Polygon",
                        "coordinates": coord
                    }),label_class=categories_by_id[label_element["category_id"]]["name"],
                                     bbox_type="Horizontal BBox", date_time="")
                object_labels.append(labels)
            training_type = categorize_string(os.path.basename(os.path.dirname(image_json["coco_url"])))

            numbers_of_labels = len(object_labels)
            td = AI_EOTrainingData(id=str(image_json["id"]),type="AI_EOTrainingData",data_sources=[""],
                                 dataset_id=dataset_id, training_type=training_type,
                                number_of_labels=numbers_of_labels, labels=object_labels,
                                date_time=[image_json["date_captured"].replace(' ', 'T')],extent=None, data_URL=[image_json["coco_url"]])
            td_list.append(td)

    # end of timer
    end_time = time.time()
    # Calculation of total and average time
    total_time = end_time - start_time
    average_time = total_time / amount_of_trainingdata

    print(f"Total time for {amount_of_trainingdata} training isntances: {total_time:.5f} seconds")
    print(f"Average time per training instance: {average_time * 60:.5f} ms")

    dataset = EOTrainingDataset(id=str(dataset_id),
        type="AI_EOTrainingDataset",
        name=dataset_name,
        description=dataset_description,
        tasks=[AI_EOTask(task_type="Object Detection",
                      id=str(dataset_id) + "_task",
                      dataset_id=str(dataset_id),
                      type='AI_EOTask',
                      description="Structural high-resolution satellite image indexing")],
        version=dataset_version,
        amount_of_training_data=amount_of_trainingdata,
        created_time=created_time,
        updated_time=updated_time,
        providers=providers,
        keywords=keywords,
        classes=classes,
        number_of_classes=number_of_classes,
        license=dataset_licenses,
        data=td_list,
        extent=None
    )
    # write to json
    write_to_json(dataset, output_json_path)


