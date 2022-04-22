# ------------------------------------------------------------------------------
#
# Project: pytdml
#
# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------


from dataclasses import dataclass, field
from typing import List, Union

import geojson
from geojson import Point, LineString, Polygon
from pytdml.type.basic_types import Label, TrainingData, TrainingDataset, TrainingDataQuality, Task, \
    MetricsInLiterature, \
    KeyValuePair, Labeling


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
            'class': self.label_class
        }

    @staticmethod
    def from_dict(json_dict: dict):
        if json_dict.__contains__("class") and json_dict["type"] == "SceneLabel":
            label = SceneLabel(label_class=json_dict["class"])
            if json_dict.__contains__("isNegative"):
                label.is_negative = json_dict["isNegative"]
            return label
        else:
            raise ValueError("The given json_dict is not a valid SceneLabel")


@dataclass
class ObjectLabel(Label):
    """
    Extended label type for object level training data
    """
    object: Union[Point, LineString, Polygon] = field(default=None)
    label_class: str = field(default=None)
    geometry_type: str = field(default=None)
    is_difficultly_detectable: bool = field(default=None)

    def to_dict(self):
        return {
            "type": "ObjectLabel",
            "isNegative": self.is_negative,
            'object': self.object,
            'geometryType': self.geometry_type,
            'class': self.label_class,
            'isDiffDetectable': self.is_difficultly_detectable
        }

    @staticmethod
    def from_dict(json_dict: dict):
        if json_dict.__contains__("object") and json_dict["type"] == "ObjectLabel":
            label = ObjectLabel(object=geojson.loads(json_dict["object"].__str__().replace("'", "\"")))
            if json_dict.__contains__("isNegative"):
                label.is_negative = json_dict["isNegative"]
            if json_dict.__contains__("geometryType"):
                label.geometry_type = json_dict["geometryType"]
            if json_dict.__contains__("class"):
                label.label_class = json_dict["class"]
            if json_dict.__contains__("isDiffDetectable"):
                label.is_difficultly_detectable = json_dict["isDiffDetectable"]
            return label
        else:
            raise ValueError("The given json_dict is not a valid ObjectLabel")


@dataclass
class PixelLabel(Label):
    """
    Extended label type for pixel level training data
    """
    image_url: Union[str, List[str]] = field(default=None)

    def to_dict(self):
        return {
            "type": "PixelLabel",
            "isNegative": self.is_negative,
            'imageUrl': self.image_url
        }

    @staticmethod
    def from_dict(json_dict: dict):
        if json_dict.__contains__("imageUrl") and json_dict["type"] == "PixelLabel":
            label = PixelLabel(image_url=json_dict["imageUrl"])
            if json_dict.__contains__("isNegative"):
                label.is_negative = json_dict["isNegative"]
            return label
        else:
            raise ValueError("The given json_dict is not a valid PixelLabel")


@dataclass
class EODataSource:
    """
    EO data source type
    """
    id: str
    data_type: str = field(default=None)
    citation: str = field(default=None)
    platform: str = field(default=None)
    sensor: str = field(default=None)
    resolution: str = field(default=None)

    def to_dict(self):
        return {
            'id': self.id,
            'dataType': self.data_type,
            'citation': self.citation,
            'platform': self.platform,
            'sensor': self.sensor,
            'resolution': self.resolution
        }

    @staticmethod
    def from_dict(json_dict: dict):
        if json_dict.__contains__("id"):
            data_source = EODataSource(json_dict["id"])
            if json_dict.__contains__("dataType"):
                data_source.data_type = json_dict["dataType"]
            if json_dict.__contains__("citation"):
                data_source.citation = json_dict["citation"]
            if json_dict.__contains__("platform"):
                data_source.platform = json_dict["platform"]
            if json_dict.__contains__("sensor"):
                data_source.sensor = json_dict["sensor"]
            if json_dict.__contains__("resolution"):
                data_source.resolution = json_dict["resolution"]
            return data_source
        else:
            raise ValueError("The given json_dict is not a valid EODataSource")


@dataclass
class EOTask(Task):
    """
    Extended task type for EO training data
    """
    task_type: str = field(default=None)

    def to_dict(self):
        return {
            "type": "EOTask",
            "description": self.description,
            "taskType": self.task_type
        }

    @staticmethod
    def from_dict(json_dict: dict):
        if json_dict.__contains__("type") and json_dict["type"] == "EOTask":
            task = EOTask()
            if json_dict.__contains__("description"):
                task.description = json_dict["description"]
            if json_dict.__contains__("taskType"):
                task.task_type = json_dict["taskType"]
            return task
        else:
            raise ValueError("Invalid EOTask dict")


@dataclass
class EOTrainingDataQuality(TrainingDataQuality):
    """
    Extended quality type for EO training data
    """

    def to_dict(self):
        return {
            "type": "EOTrainingDataQuality",
        }

    @staticmethod
    def from_dict(json_dict: dict):
        if json_dict["type"] == "EOTrainingDataQuality":
            tdq = EOTrainingDataQuality()
            return tdq
        else:
            raise ValueError("Invalid EOTrainingDataQuality dict")


@dataclass
class EOTrainingData(TrainingData):
    """
    Extended training data type for EO training data
    """
    extent: List[float] = field(default=None)
    date_time: Union[str, List[str]] = field(default=None)
    data_url: Union[str, List[str]] = field(default=None)

    def to_dict(self):
        return {
            "type": "EOTrainingData",
            "id": self.id,
            "trainingType": self.training_type,
            "numberOfLabels": self.number_of_Labels,
            "labels": [label.to_dict() for label in self.labels],
            'extent': self.extent,
            'dateTime': self.date_time,
            'dataUrl': self.data_url
        }

    @staticmethod
    def from_dict(json_dict: dict):
        if json_dict.__contains__("id") and json_dict["type"] == "EOTrainingData":
            training_data = EOTrainingData(json_dict["id"])
            if json_dict.__contains__("trainingType"):
                training_data.training_type = json_dict["trainingType"]
            if json_dict.__contains__("numberOfLabels"):
                training_data.number_of_Labels = json_dict["numberOfLabels"]
            if json_dict.__contains__("labels"):
                # Different type of labels
                if json_dict["labels"][0]["type"] == "SceneLabel":
                    training_data.labels = [SceneLabel.from_dict(label) for label in json_dict["labels"]]
                elif json_dict["labels"][0]["type"] == "ObjectLabel":
                    training_data.labels = [ObjectLabel.from_dict(label) for label in json_dict["labels"]]
                elif json_dict["labels"][0]["type"] == "PixelLabel":
                    training_data.labels = [PixelLabel.from_dict(label) for label in json_dict["labels"]]
            if json_dict.__contains__("extent"):
                training_data.extent = json_dict["extent"]
            if json_dict.__contains__("dateTime"):
                training_data.date_time = json_dict["dateTime"]
            if json_dict.__contains__("dataUrl"):
                training_data.data_url = json_dict["dataUrl"]
            return training_data
        else:
            raise ValueError("The given json_dict is not a valid EOTrainingData")


@dataclass
class EOTrainingDataset(TrainingDataset):
    """
    Extended training dataset type for EO training dataset
    """
    quality: TrainingDataQuality = field(default=EOTrainingDataQuality())
    extent: List[float] = field(default=None)
    data_sources: List[EODataSource] = field(default=None)
    bands: List[str] = field(default=None)
    image_size: str = field(default=None)

    def to_dict(self):
        return {
            "type": "EOTrainingDataset",
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
            'extent': self.extent,
            'dataSources': [ds.to_dict() for ds in self.data_sources],
            'bands': self.bands,
            'imageSize': self.image_size,
            "tasks": [t.to_dict() for t in self.tasks],
            "labelings": [labeling.to_dict() for labeling in self.labelings],
            "quality": self.quality.to_dict(),
            "data": [d.to_dict() for d in self.data]
        }

    @staticmethod
    def from_dict(json_dict: dict):
        if json_dict.__contains__("id") and json_dict.__contains__("name") \
                and json_dict.__contains__("description") and json_dict["type"] == "EOTrainingDataset":
            td = EOTrainingDataset(json_dict["id"], json_dict["name"], json_dict["description"])
            if json_dict.__contains__("name"):
                td.name = json_dict["name"]
            if json_dict.__contains__("description"):
                td.description = json_dict["description"]
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
            if json_dict.__contains__("extent"):
                td.extent = json_dict["extent"]
            if json_dict.__contains__("dataSources"):
                td.data_sources = [EODataSource.from_dict(ds) for ds in json_dict["dataSources"]]
            if json_dict.__contains__("bands"):
                td.bands = json_dict["bands"]
            if json_dict.__contains__("imageSize"):
                td.image_size = json_dict["imageSize"]
            if json_dict.__contains__("tasks"):
                td.tasks = [EOTask.from_dict(t) for t in json_dict["tasks"]]
            if json_dict.__contains__("labelings"):
                td.labelings = [Labeling.from_dict(labeling) for labeling in json_dict["labelings"]]
            if json_dict.__contains__("quality"):
                td.quality = EOTrainingDataQuality.from_dict(json_dict["quality"])
            if json_dict.__contains__("data"):
                td.data = [EOTrainingData.from_dict(data) for data in json_dict["data"]]
            return td
        else:
            raise ValueError("Invalid EOTrainingDataset dict")
