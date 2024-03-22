
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
import json
from dataclasses import dataclass, field, asdict
from typing import List, Union

import geojson
from geojson import Feature
from pytdml.type.basic_types_old import Label, TrainingData, TrainingDataset, DataQuality, Task, \
    MetricsInLiterature, \
    KeyValuePair, Labeling, Changeset, Scope


@dataclass
class SceneLabel(Label):
    """
    Extended label type for scene level training data
    """
    label_class: str = field(default=None)

    def to_dict(self):
        return {
            "type": "SceneLabel",
            "isNegative": self.is_negative,
            "confidence": self.confidence,
            'class': self.label_class
        }

    @staticmethod
    def from_dict(json_dict: dict):
        if json_dict.__contains__("class") and json_dict["type"] == "SceneLabel":
            label = SceneLabel(label_class=json_dict["class"])
            if json_dict.__contains__("isNegative"):
                label.is_negative = json_dict["isNegative"]
            if json_dict.__contains__("confidence"):
                label.confidence = json_dict["confidence"]
            return label
        else:
            raise ValueError("The given json_dict is not a valid SceneLabel")


@dataclass
class ObjectLabel(Label):
    """
    Extended label type for object level training data
    """
    object: Feature = field(default=None)
    label_class: str = field(default=None)
    bbox_type: str = field(default=None)
    is_difficultly_detectable: bool = field(default=None)
    date_time: str = field(default=None)

    def to_dict(self):
        return {
            "type": "ObjectLabel",
            "isNegative": self.is_negative,
            "confidence": self.confidence,
            'object': self.object,
            'bboxType': self.bbox_type,
            'class': self.label_class,
            'isDiffDetectable': self.is_difficultly_detectable,
            'dateTime': self.date_time
        }

    @staticmethod
    def from_dict(json_dict: dict):
        if json_dict.__contains__("object") and json_dict["type"] == "ObjectLabel":
            # label = ObjectLabel(object=geojson.loads(json_dict["object"].__str__().replace("'", "\"")))
            label = ObjectLabel(object=geojson.loads(json.dumps(json_dict["object"])))

            if json_dict.__contains__("isNegative"):
                label.is_negative = json_dict["isNegative"]
            if json_dict.__contains__("confidence"):
                label.confidence = json_dict["confidence"]
            if json_dict.__contains__("bboxType"):
                label.bbox_type = json_dict["bboxType"]
            if json_dict.__contains__("class"):
                label.label_class = json_dict["class"]
            if json_dict.__contains__("isDiffDetectable"):
                label.is_difficultly_detectable = json_dict["isDiffDetectable"]
            if json_dict.__contains__("dateTime"):
                label.date_time = json_dict["dateTime"]
            return label
        else:
            raise ValueError("The given json_dict is not a valid ObjectLabel")


@dataclass
class PixelLabel(Label):
    """
    Extended label type for pixel level training data
    """
    image_url: Union[str, List[str]] = field(default=None)
    image_format: Union[str, List[str]] = field(default=None)

    def to_dict(self):
        return {
            "type": "PixelLabel",
            "isNegative": self.is_negative,
            "confidence": self.confidence,
            "imageURL": self.image_url,
            "imageFormat": self.image_format
        }

    @staticmethod
    def from_dict(json_dict: dict):
        if json_dict.__contains__("imageURL") and json_dict.__contains__("imageFormat") \
                and json_dict["type"] == "PixelLabel":
            label = PixelLabel(image_url=json_dict["imageURL"], image_format=json_dict["imageFormat"])
            if json_dict.__contains__("isNegative"):
                label.is_negative = json_dict["isNegative"]
            if json_dict.__contains__("confidence"):
                label.confidence = json_dict["confidence"]
            return label
        else:
            raise ValueError("The given json_dict is not a valid PixelLabel")


# @dataclass
# class EODataSource:
#     """
#     EO data source type
#     """
#     id: str
#     data_type: str = field(default=None)
#     citation: str = field(default=None)
#     platform: str = field(default=None)
#     sensor: str = field(default=None)
#     resolution: str = field(default=None)
#     format: str = field(default=None)
#
#     def to_dict(self):
#         return {
#             'id': self.id,
#             'dataType': self.data_type,
#             'citation': self.citation,
#             'platform': self.platform,
#             'sensor': self.sensor,
#             'resolution': self.resolution,
#             'format': self.format
#         }
#
#     @staticmethod
#     def from_dict(json_dict: dict):
#         if json_dict.__contains__("id"):
#             data_source = EODataSource(json_dict["id"])
#             if json_dict.__contains__("dataType"):
#                 data_source.data_type = json_dict["dataType"]
#             if json_dict.__contains__("citation"):
#                 data_source.citation = json_dict["citation"]
#             if json_dict.__contains__("platform"):
#                 data_source.platform = json_dict["platform"]
#             if json_dict.__contains__("sensor"):
#                 data_source.sensor = json_dict["sensor"]
#             if json_dict.__contains__("resolution"):
#                 data_source.resolution = json_dict["resolution"]
#             if json_dict.__contains__("format"):
#                     data_source.resolution = json_dict["format"]
#             return data_source
#         else:
#             raise ValueError("The given json_dict is not a valid EODataSource")


@dataclass
class EOTask(Task):
    """
    Extended task type for EO training data
    """
    task_type: str = field(default=None)

    def to_dict(self):
        return {
            "type": "EOTask",
            "id": self.id,
            "datasetId": self.dataset_id,
            "description": self.description,
            "taskType": self.task_type
        }

    @staticmethod
    def from_dict(json_dict: dict):
        if json_dict.__contains__("id") and json_dict.__contains__("type") and json_dict["type"] == "EOTask":
            task = EOTask(json_dict["id"])
            if json_dict.__contains__("datasetId"):
                task.dataset_id = json_dict["datasetId"]
            if json_dict.__contains__("description"):
                task.description = json_dict["description"]
            if json_dict.__contains__("taskType"):
                task.task_type = json_dict["taskType"]
            return task
        else:
            raise ValueError("The given json_dict is not a valid EOTask")


@dataclass
class EOTrainingData(TrainingData):
    """
    Extended training data type for EO training data
    """
    extent: List[float] = field(default_factory=list)
    date_time: Union[str, List[str]] = field(default=None)
    data_url: Union[str, List[str]] = field(default=None)

    def to_dict(self):
        return {
            "type": "AI_EOTrainingData",
            "id": self.id,
            "trainingType": self.training_type,
            "numberOfLabels": self.number_of_labels,
            "dataSources": self.data_sources,
            'extent': self.extent,
            'dateTime': self.date_time,
            'dataURL': self.data_url,
            "quality": self.quality.to_dict() if self.quality is not None else None,
            "labeling": [ll.to_dict() for ll in self.labeling],
            "labels": [label.to_dict() for label in self.labels]
        }

    @staticmethod
    def from_dict(json_dict: dict):
        if json_dict.__contains__("id") and json_dict["type"] == "EOTrainingData":
            training_data = EOTrainingData(json_dict["id"])
            if json_dict.__contains__("trainingType"):
                training_data.training_type = json_dict["trainingType"]
            if json_dict.__contains__("numberOfLabels"):
                training_data.number_of_labels = json_dict["numberOfLabels"]
            if json_dict.__contains__("dataSources"):
                training_data.data_sources = json_dict["dataSources"]
            if json_dict.__contains__("quality"):
                training_data.quality = DataQuality.from_dict(json_dict["quality"])
            if json_dict.__contains__("labeling"):
                training_data.labeling = [Labeling.from_dict(ll) for ll in json_dict["labeling"]]
            if json_dict.__contains__("labels"):
                # Different type of labels
                if len(json_dict["labels"]) > 0:
                    if json_dict["labels"][0]["type"] == "SceneLabel":
                        training_data.labels = [SceneLabel.from_dict(label) for label in json_dict["labels"]]
                    elif json_dict["labels"][0]["type"] == "ObjectLabel":
                        training_data.labels = [ObjectLabel.from_dict(label) for label in json_dict["labels"]]
                    elif json_dict["labels"][0]["type"] == "PixelLabel":
                        training_data.labels = [PixelLabel.from_dict(label) for label in json_dict["labels"]]
                else:
                    training_data.labels = []
            if json_dict.__contains__("extent"):
                training_data.extent = json_dict["extent"]
            if json_dict.__contains__("dateTime"):
                training_data.date_time = json_dict["dateTime"]
            if json_dict.__contains__("dataURL"):
                training_data.data_url = json_dict["dataURL"]
            if json_dict.__contains__("datasetId"):
                training_data.dataset_id = json_dict["datasetId"]
            return training_data
        else:
            raise ValueError("The given json_dict is not a valid EOTrainingData")


@dataclass
class EOTrainingDataset(TrainingDataset):
    """
    Extended training dataset type for EO training dataset
    """
    extent: List[float] = field(default_factory=list)
    bands: List[str] = field(default_factory=list)
    image_size: str = field(default=None)

    def to_dict(self):
        return {
            "type": "EOTrainingDataset",
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "license": self.license,
            "doi": self.doi,
            "scope": self.scope.to_dict() if self.scope is not None else None,
            "version": self.version,
            "amountOfTrainingData": self.amount_of_training_data,
            "createdTime": self.created_time,
            "updatedTime": self.updated_time,
            "providers": self.providers,
            "keywords": self.keywords,
            "metricsInLIT": [m.to_dict() for m in self.metrics_in_literature],
            "statisticsInfo": [s.to_dict() for s in self.statistics_info],
            "numberOfClasses": self.number_of_classes,
            "classificationSchema": self.classification_schema,
            "classes": self.classes,
            "tasks": [t.to_dict() for t in self.tasks],
            'extent': self.extent,
            'bands': self.bands,
            'imageSize': self.image_size,
            "labeling": [labeling.to_dict() for labeling in self.labeling],
            "quality": self.quality.to_dict() if self.quality is not None else None,
            "changesets": [changeset.to_dict() for changeset in self.changesets],
            "data": [d.to_dict() for d in self.data]
        }

    @staticmethod
    def from_dict(json_dict: dict):
        if json_dict.__contains__("id") and json_dict.__contains__("name") \
                and json_dict.__contains__("description") \
                and json_dict.__contains__("license") \
                and json_dict["type"] == "EOTrainingDataset":

            td = EOTrainingDataset(json_dict["id"], json_dict["name"], json_dict["description"], json_dict["license"])
            if json_dict.__contains__("doi"):
                td.doi = json_dict["doi"]
            if json_dict.__contains__("scope"):
                td.scope = Scope.from_dict(json_dict["scope"])
            if json_dict.__contains__("version"):
                td.version = json_dict["version"]
            if json_dict.__contains__("amountOfTrainingData"):
                td.amount_of_training_data = json_dict["amountOfTrainingData"]
            if json_dict.__contains__("createdTime"):
                td.created_time = json_dict["createdTime"]
            if json_dict.__contains__("updatedTime"):
                td.updated_time = json_dict["updatedTime"]
            if json_dict.__contains__("providers"):
                td.providers = json_dict["providers"]
            if json_dict.__contains__("keywords"):
                td.keywords = json_dict["keywords"]
            if json_dict.__contains__("metricsInLIT"):
                td.metrics_in_literature = [MetricsInLiterature.from_dict(m) for m in
                                            json_dict["metricsInLIT"]]
            if json_dict.__contains__("statisticsInfo"):
                td.statistics_info = [KeyValuePair.from_dict(s) for s in json_dict["statisticsInfo"]]
            if json_dict.__contains__("numberOfClasses"):
                td.number_of_classes = json_dict["numberOfClasses"]
            if json_dict.__contains__("classificationSchema"):
                td.classification_schema = json_dict["classificationSchema"]
            if json_dict.__contains__("classes"):
                td.classes = json_dict["classes"]
            if json_dict.__contains__("tasks"):
                td.tasks = [EOTask.from_dict(t) for t in json_dict["tasks"]]
            if json_dict.__contains__("labeling"):
                td.labeling = [Labeling.from_dict(labeling) for labeling in json_dict["labeling"]]
            if json_dict.__contains__("quality"):
                td.quality = DataQuality.from_dict(json_dict["quality"])
            if json_dict.__contains__("changesets"):
                td.changesets = [Changeset.from_dict(changeset) for changeset in json_dict["changesets"]]
            if json_dict.__contains__("data"):
                td.data = [EOTrainingData.from_dict(data) for data in json_dict["data"]]
            if json_dict.__contains__("extent"):
                td.extent = json_dict["extent"]
            if json_dict.__contains__("bands"):
                td.bands = json_dict["bands"]
            if json_dict.__contains__("imageSize"):
                td.image_size = json_dict["imageSize"]
            return td
        else:
            raise ValueError("The given json_dict is not a valid EOTrainingDataset")
