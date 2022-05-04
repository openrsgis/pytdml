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
from dataclasses import dataclass, field
from typing import List, Union


@dataclass
class KeyValuePair:
    """
    Key/Value pair type
    """
    key: str
    value: Union[str, int, float, bool, None]

    def to_dict(self):
        return {self.key: self.value}

    @staticmethod
    def from_dict(json_dict: dict):
        if len(json_dict.keys()) == 1:
            for key in json_dict.keys():
                return KeyValuePair(key, json_dict[key])
        else:
            raise ValueError("The given json_dict is not a valid KeyValuePair")


@dataclass
class MetricsInLiterature:
    """
    Metrics in literature type
    """
    doi: str
    algorithm: str = field(default=None)
    metrics: List[KeyValuePair] = field(default_factory=list)

    def to_dict(self):
        return {
            "doi": self.doi,
            "algorithm": self.algorithm,
            "metrics": [m.to_dict() for m in self.metrics]
        }

    @staticmethod
    def from_dict(json_dict: dict):
        if json_dict.__contains__("doi"):
            metrics_in_literature = MetricsInLiterature(json_dict["doi"])
            if json_dict.__contains__("algorithm"):
                metrics_in_literature.algorithm = json_dict["algorithm"]
            if json_dict.__contains__("metrics"):
                metrics_in_literature.metrics = [KeyValuePair.from_dict(m) for m in json_dict["metrics"]]
            return metrics_in_literature
        else:
            raise ValueError("The given json_dict is not a valid MetricsInLiterature")


@dataclass
class Task:
    """
    Basic task type
    """
    description: str = field(default=None)

    def to_dict(self):
        return {
            "type": "Task",
            "description": self.description
        }

    @staticmethod
    def from_dict(json_dict: dict):
        if json_dict["type"] == "Task":
            task = Task()
            if json_dict.__contains__("type"):
                task.description = json_dict["description"]
            return task
        else:
            raise ValueError("The given json_dict is not a valid Task")


@dataclass
class Labeler:
    """
    Labeler type
    """
    id: str
    name: str = field(default=None)

    def to_dict(self):
        return {
            "type": "Labeler",
            "id": self.id,
            "name": self.name
        }

    @staticmethod
    def from_dict(json_dict: dict):
        if json_dict.__contains__("id") and json_dict["type"] == "Labeler":
            labeler = Labeler(json_dict["id"])
            if json_dict.__contains__("name"):
                labeler.name = json_dict["name"]
            return labeler
        else:
            raise ValueError("The given json_dict is not a valid Labeler")


@dataclass
class LabelingProcedure:
    """
    Labeling procedure type
    """
    id: str
    method: str = field(default=None)
    tool: str = field(default=None)

    def to_dict(self):
        return {
            "type": "LabelingProcedure",
            "id": self.id,
            "method": self.method,
            "tool": self.tool
        }

    @staticmethod
    def from_dict(json_dict: dict):
        if json_dict.__contains__("id") and json_dict["type"] == "LabelingProcedure":
            labeling_procedure = LabelingProcedure(json_dict["id"])
            if json_dict.__contains__("method"):
                labeling_procedure.method = json_dict["method"]
            if json_dict.__contains__("tool"):
                labeling_procedure.tool = json_dict["tool"]
            return labeling_procedure
        else:
            raise ValueError("The given json_dict is not a valid LabelingProcedure")


@dataclass
class Labeling:
    """
    Labeling type
    """
    id: str
    labelers: List[Labeler] = field(default_factory=list)
    procedure: LabelingProcedure = field(default=LabelingProcedure(""))

    def to_dict(self):
        return {
            "type": "Labeling",
            "id": self.id,
            "labelers": [labeler.to_dict() for labeler in self.labelers],
            "procedure": self.procedure.to_dict()
        }

    @staticmethod
    def from_dict(json_dict: dict):
        if json_dict.__contains__("id") and json_dict["type"] == "Labeling":
            labeling = Labeling(json_dict["id"])
            if json_dict.__contains__("labelers"):
                labeling.labelers = [Labeler.from_dict(labeler) for labeler in json_dict["labelers"]]
            if json_dict.__contains__("procedure"):
                labeling.procedure = LabelingProcedure.from_dict(json_dict["procedure"])
            return labeling
        else:
            raise ValueError("The given json_dict is not a valid Labeling")


@dataclass
class TrainingDataQuality:
    """
    Basic training data quality type
    """

    def to_dict(self):
        return {
            "type": "TrainingDataQuality"
        }

    @staticmethod
    def from_dict(json_dict: dict):
        if json_dict["type"] == "TrainingDataQuality":
            return TrainingDataQuality()
        else:
            raise ValueError("The given json_dict is not a valid TrainingDataQuality")


@dataclass
class Label:
    """
    Basic label type
    """
    is_negative: bool = field(default=None)

    def to_dict(self):
        return {
            "type": "Label",
            "isNegative": self.is_negative
        }

    @staticmethod
    def from_dict(json_dict: dict):
        if json_dict["type"] == "Label":
            label = Label()
            if json_dict.__contains__("isNegative"):
                label.is_negative = json_dict["isNegative"]
            return label
        else:
            raise ValueError("The given json_dict is not a valid Label")


@dataclass
class TrainingData:
    """
    Basic training data type
    """
    id: str
    training_type: str = field(default=None)
    number_of_Labels: int = field(default=None)
    labels: List[Label] = field(default_factory=list)

    def get_labels(self) -> List[Label]:
        """
        Returns the labels of the training data
        """
        return self.labels

    def to_dict(self):
        return {
            "type": "TrainingData",
            "id": self.id,
            "trainingType": self.training_type,
            "numberOfLabels": self.number_of_Labels,
            "labels": [label.to_dict() for label in self.labels]
        }

    @staticmethod
    def from_dict(json_dict: dict):
        if json_dict.__contains__("id") and json_dict["type"] == "TrainingData":
            training_data = TrainingData(json_dict["id"])
            if json_dict.__contains__("trainingType"):
                training_data.training_type = json_dict["trainingType"]
            if json_dict.__contains__("numberOfLabels"):
                training_data.number_of_Labels = json_dict["numberOfLabels"]
            if json_dict.__contains__("labels"):
                training_data.labels = [Label.from_dict(label) for label in json_dict["labels"]]
            return training_data
        else:
            raise ValueError("The given json_dict is not a valid TrainingData")


@dataclass
class TrainingDataset:
    """
    Basic training dataset type
    """
    id: str
    name: str
    description: str
    version: str = field(default=None)
    amount_of_training_data: int = field(default=None)
    created_time: str = field(default=None)
    updated_time: str = field(default=None)
    license: str = field(default=None)
    providers: List[str] = field(default=None)
    keywords: List[str] = field(default=None)
    metrics_in_literature: List[MetricsInLiterature] = field(default_factory=list)
    statistics_info: List[KeyValuePair] = field(default_factory=list)
    number_of_classes: int = field(default=None)
    classification_schema: str = field(default=None)
    classes: Union[List[KeyValuePair], List[str]] = field(default=None)
    tasks: List[Task] = field(default_factory=list)
    labelings: List[Labeling] = field(default_factory=list)
    quality: TrainingDataQuality = field(default=TrainingDataQuality())
    data: List[TrainingData] = field(default_factory=list)

    def get_training_data(self) -> List[TrainingData]:
        """
        Return the training data of the training dataset
        """
        return self.data

    def get_training_data_by_id(self, _id: str) -> TrainingData:
        """
        Get training data of the training dataset by id
        """
        for data in self.data:
            if data.id == _id:
                return data

    def get_labelings(self) -> List[Labeling]:
        """
        Return the labelings of the training dataset
        """
        return self.labelings

    def get_tasks(self) -> List[Task]:
        """
        Return the tasks of the training dataset
        """
        return self.tasks

    def get_quality(self) -> TrainingDataQuality:
        """
        Return the quality of the training dataset
        """
        return self.quality

    def to_dict(self):
        return {
            "type": "TrainingDataset",
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "amountOfTrainingData": self.amount_of_training_data,
            "createdTime": self.created_time,
            "updatedTime": self.updated_time,
            "license": self.license,
            "providers": self.providers,
            "keywords": self.keywords,
            "metricsInLIT": [m.to_dict() for m in self.metrics_in_literature],
            "statisticsInfo": [s.to_dict() for s in self.statistics_info],
            "numberOfClasses": self.number_of_classes,
            "classificationSchema": self.classification_schema,
            "classes": self.classes,
            "tasks": [t.to_dict() for t in self.tasks],
            "labelings": [labeling.to_dict() for labeling in self.labelings],
            "quality": self.quality.to_dict(),
            "data": [d.to_dict() for d in self.data]
        }

    @staticmethod
    def from_dict(json_dict: dict):
        if json_dict.__contains__("id") and json_dict.__contains__("name") \
                and json_dict.__contains__("description") and json_dict["type"] == "TrainingDataset":
            td = TrainingDataset(json_dict["id"], json_dict["name"], json_dict["description"])
            if json_dict.__contains__("version"):
                td.version = json_dict["version"]
            if json_dict.__contains__("amountOfTrainingData"):
                td.amount_of_training_data = json_dict["amountOfTrainingData"]
            if json_dict.__contains__("createdTime"):
                td.created_time = json_dict["createdTime"]
            if json_dict.__contains__("updatedTime"):
                td.updated_time = json_dict["updatedTime"]
            if json_dict.__contains__("license"):
                td.license = json_dict["license"]
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
                td.tasks = [Task.from_dict(t) for t in json_dict["tasks"]]
            if json_dict.__contains__("labelings"):
                td.labelings = [Labeling.from_dict(labeling) for labeling in json_dict["labelings"]]
            if json_dict.__contains__("quality"):
                td.quality = TrainingDataQuality.from_dict(json_dict["quality"])
            if json_dict.__contains__("data"):
                td.data = [TrainingData.from_dict(data) for data in json_dict["data"]]
            return td
        else:
            raise ValueError("The given json_dict is not a valid TrainingDataset")
