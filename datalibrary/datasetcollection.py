# ------------------------------------------------------------------------------
#
# Project: pytdml
# Authors: Shuaiqi Liu
# Created: 2023-02-04
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
import os
from dataclasses import dataclass, field
from typing import List
from urllib.parse import urlparse

import pytdml.utils as utils
from pytdml.io import internal, read_from_json, parse_json
from pytdml.type.extended_types import EOTrainingDataset, AI_EOTask


@dataclass
class EOTrainingDatasetCollection:
    eo_training_datasets_list: List[dict] = field(
        default_factory=internal.datasets_load, init=False
    )
    task_list: List[str] = field(default_factory=internal.task_load, init=False)

    def dataset_list(self, task_type=None, classes=None):
        """
        Args:
            task_type (str): Argument must be one element of ['Scene Classification', 'Objection Detection',
            'Semantic Segmentation', 'Change Detection', '3D Model Reconstruction'].

            classes(List): An array of categories under the task.
        Returns:
            List(dict):
        """
        if task_type is not None:
            if task_type not in self.task_list:
                raise ValueError("task must be one of {}".format(str(self.task_list)))
            print("with Training task: " + task_type)

        if classes is not None:
            if not isinstance(classes, list):
                raise ValueError("selected classes must be a list!")
            print("with selected classes: " + str(classes))

        training_datasets = utils.datasets_list(
            self.eo_training_datasets_list, task_type, classes
        )

        print(
            "Here are {} training datasets in library:".format(len(training_datasets))
        )
        if len(training_datasets) == 0:
            return []
        for item in training_datasets:
            print("[{}]: {}".format(str(item["name"]), str(item["description"])))
            print("classes: {}".format(str(item["classes"])))
            print("bands: {}".format(str(item["band_size"])))
            print("imageSize: {}".format(str(item["image_size"])))
            print("\n")
        return training_datasets

    def __getitem__(self, dataset_name) -> EOTrainingDataset:
        """
        Args:
            dataset_name (str): dataset name.

        Returns:
            EOTrainingDataset: Training Dataset Markup Language of the dataset .
        """
        datasets_item = list(
            filter(
                lambda item: item["name"] == dataset_name,
                self.eo_training_datasets_list,
            )
        )

        if len(datasets_item) == 0:
            raise ValueError(
                "dataset {} is not in Collections, Please check your input parameters!"
            )

        datasets_tdml = internal.read_from_server(datasets_item[0]["name"])
        return datasets_tdml

    def fetch_tdml(self, url):
        """
        Args:
            url (str): tdml url.

        Returns:
            EOTrainingDataset: Training Dataset Markup Language of the dataset .
        """
        # schema = internal.schema_load()
        if os.path.exists(url):
            dataset_td_encode = read_from_json(url)

            # If the URL points to a local file, read the JSON content from the file
            return dataset_td_encode

        elif urlparse(url).scheme in ["http", "https"] and urlparse(url).netloc != "":
            try:
                import requests
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "Failed to import requests, please install the library first"
                )
            response = requests.get(url).json()
            dataset_td_encode = parse_json(response)

            # If the URL is an HTTP/HTTPS link, send a GET request and retrieve the JSON response
            return dataset_td_encode
        else:
            raise ValueError("Invalid url")

    def training_data_collection(self, task, dataset_list, classes):
        """
        Collects and merges training data from multiple datasets for a given task.

        Args:
            task (str): The type of Geo AI task. Supported tasks are "Change Detection", "3D Model Reconstruction",
                        "object_detection", "scene_classification", and "semantic_segmentation".
            dataset_list (list): A list of EOTrainingDataset instances containing the training datasets.
            classes (list): A list of class labels to filter the training data.

        Returns:
            EOTrainingDataset: A merged EOTrainingDataset instance containing the collected training data.

        Raises:
            ValueError: If the task is not supported or the dataset_list contains different Geo AI task types.

        """

        # Check if the task is "Change Detection" or "3D Model Reconstruction"
        if task in [Task.change_detection, Task.model_3d_reconstruction]:
            if len(dataset_list) > 1:
                raise ValueError(f"{task} types not supported for use across datasets")

        # Check if dataset_list is a list and contains instances of EOTrainingDataset
        if isinstance(dataset_list, list) and all(
            isinstance(ele, EOTrainingDataset) for ele in dataset_list
        ):
            task_set = set(item.tasks[0].task_type for item in dataset_list)

            # Check if all datasets have the same task type
            if not (len(task_set) == 1):
                raise ValueError("argument datasets must be the same Geo AI task type.")

            merged_td_list = []

            for dataset_item in dataset_list:
                if len(dataset_item.bands) > 4:
                    raise ValueError(
                        "Hyper spectral sample dataset: {} with {} bands does not support cross-dataset use.".format(
                            dataset_item.name, len(dataset_item.bands)
                        )
                    )
                # Check the validity of classes for the dataset_item
                classes = _check_validity_class(classes, dataset_item)

                if task == Task.object_detection:
                    merged_td_list.extend(
                        [
                            item
                            for item in dataset_item.data
                            if utils.load_data_list_(item.labels, classes)
                        ]
                    )

                if task == Task.scene_classification:
                    merged_td_list.extend(
                        [
                            item
                            for item in dataset_item.data
                            if item.labels[0].label_class in classes
                        ]
                    )

                if task == Task.semantic_segmentation:
                    merged_td_list.extend([item for item in dataset_item.data])
            # generate an new EOTrainingDataset
            new_id, new_name, new_description, new_classes, new_bands = (
                utils.generate_new_tdml(dataset_list, classes)
            )
            return EOTrainingDataset(
                id=new_id,
                name=new_name,
                description=new_description,
                tasks=[
                    AI_EOTask(id=new_id + " Task", type="AI_EOTask", task_type=task)
                ],
                license="",
                data=merged_td_list,
                type="AI_EOTrainingDataset",
                amount_of_training_data=len(merged_td_list),
                classes=new_classes,
                number_of_classes=len(classes),
                bands=new_bands,
            )


class Task:
    scene_classification = "Scene Classification"
    object_detection = "Object Detection"
    semantic_segmentation = "Semantic Segmentation"
    change_detection = "Change Detection"
    model_3d_reconstruction = "3D Model Reconstruction"


def _check_validity_class(classes, dataset):
    """
    Checks the validity of the provided classes against the categories of the dataset.

    Args:
        classes (list): List of classes to check.
        dataset (EOTrainingDataset): Dataset to compare the classes against.

    Returns:
        list: Validated list of classes.

    Raises:
        ValueError: If the provided classes contain elements that are not present in the dataset's categories.

    """
    if isinstance(dataset.classes[0], dict):
        if not set(classes).issubset(
            {key for item in dataset.classes for key in item.keys()}
        ):
            raise ValueError(
                "argument classes must be a subset of the categories of dataset {}".format(
                    dataset.name
                )
            )
    elif isinstance(dataset.classes[0], list):

        if not set(classes).issubset(set(dataset.classes)):
            raise ValueError(
                "argument classes must be a subset of the categories of dataset {}".format(
                    dataset.name
                )
            )
    return classes
