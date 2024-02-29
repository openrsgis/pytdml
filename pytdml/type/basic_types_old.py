
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
        # if len(json_dict.keys()) == 1:
        #     for key in json_dict.keys():
        #         return KeyValuePair(key, json_dict[key])
        if "count" in json_dict and "class" in json_dict:
            return KeyValuePair(json_dict["class"], json_dict["count"])
        else:
            raise ValueError("The given json_dict is not a valid KeyValuePair")

    def keys(self):
        pass


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
    id: str
    dataset_id: str = field(default=None)
    description: str = field(default=None)

    def to_dict(self):
        return {
            "type": "Task",
            "id": self.id,
            "datasetId": self.dataset_id,
            "description": self.description
        }

    @staticmethod
    def from_dict(json_dict: dict):
        if json_dict.__contains__("id") and json_dict["type"] == "Task":
            task = Task(json_dict["id"])
            if json_dict.__contains__("datasetId"):
                task.dataset_id = json_dict["datasetId"]
            if json_dict.__contains__("description"):
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
    methods: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "type": "LabelingProcedure",
            "id": self.id,
            "methods": self.methods,
            "tools": self.tools
        }

    @staticmethod
    def from_dict(json_dict: dict):
        if json_dict.__contains__("id") and json_dict["type"] == "LabelingProcedure":
            labeling_procedure = LabelingProcedure(json_dict["id"])
            if json_dict.__contains__("methods"):
                labeling_procedure.method = json_dict["methods"]
            if json_dict.__contains__("tools"):
                labeling_procedure.tool = json_dict["tools"]
            return labeling_procedure
        else:
            raise ValueError("The given json_dict is not a valid LabelingProcedure")


@dataclass
class Labeling:
    """
    Labeling type
    """
    id: str
    scope: object
    labelers: List[Labeler] = field(default_factory=list)
    procedure: LabelingProcedure = field(default=None)

    def to_dict(self):
        return {
            "type": "Labeling",
            "id": self.id,
            "scope": self.scope,
            "labelers": [labeler.to_dict() for labeler in self.labelers],
            "procedure": self.procedure.to_dict() if self.procedure is not None else None,
        }

    @staticmethod
    def from_dict(json_dict: dict):
        if json_dict.__contains__("id") and json_dict.__contains__("scope") and json_dict["type"] == "Labeling":
            labeling = Labeling(json_dict["id"], json_dict["scope"])
            if json_dict.__contains__("labelers"):
                labeling.labelers = [Labeler.from_dict(labeler) for labeler in json_dict["labelers"]]
            if json_dict.__contains__("procedure"):
                labeling.procedure = LabelingProcedure.from_dict(json_dict["procedure"])
            return labeling
        else:
            raise ValueError("The given json_dict is not a valid Labeling")


@dataclass
class ScopeDescription:
    """
    From ISO 19115-1 MD_ScopeDescription
    """
    attributes: List[str] = field(default_factory=list)
    features: List[str] = field(default_factory=list)
    featureInstances: List[str] = field(default_factory=list)
    attributeInstances: List[str] = field(default_factory=list)
    dataset: str = field(default=None)
    other: str = field(default=None)

    def to_dict(self):
        return {
            "attributes": self.attributes,
            "features": self.features,
            "featureInstances": self.featureInstances,
            "attributeInstances": self.attributeInstances,
            "dataset": self.dataset,
            "other": self.other
        }

    @staticmethod
    def from_dict(json_dict: dict):
        sd = ScopeDescription()
        if json_dict.__contains__("attributes"):
            sd.attributes = json_dict["attributes"]
        if json_dict.__contains__("features"):
            sd.features = json_dict["features"]
        if json_dict.__contains__("featureInstances"):
            sd.featureInstances = json_dict["featureInstances"]
        if json_dict.__contains__("attributeInstances"):
            sd.attributeInstances = json_dict["attributeInstances"]
        if json_dict.__contains__("dataset"):
            sd.dataset = json_dict["dataset"]
        if json_dict.__contains__("other"):
            sd.other = json_dict["other"]
        return sd


@dataclass
class Scope:
    """
    From ISO 19115-1 MD_Scope
    """
    level: str
    levelDescription: List[ScopeDescription] = field(default_factory=list)

    def to_dict(self):
        return {
            "level": self.level,
            "levelDescription": [ld.to_dict for ld in self.levelDescription]
        }

    @staticmethod
    def from_dict(json_dict: dict):
        if json_dict.__contains__("level"):
            scope = Scope(json_dict["level"])
            if json_dict.__contains__("levelDescription"):
                scope.levelDescription = [ScopeDescription.from_dict(ld) for ld in json_dict["levelDescription"]]
            return scope
        else:
            raise ValueError("The given json_dict is not a valid Scope")


@dataclass
class QualityElement:
    """
    From ISO 19157-1 QualityElement
    """
    type: str = field(default=None)
    measure: str = field(default=None)
    evaluation_method: str = field(default=None)
    result: str = field(default=None)

    def to_dict(self):
        return {
            "type": self.type,
            "measure": self.measure,
            "evaluationMethod": self.evaluation_method,
            "result": self.result
        }

    @staticmethod
    def from_dict(json_dict: dict):
        quality_element = QualityElement()
        if json_dict.__contains__("type"):
            quality_element.type = json_dict["type"]
        if json_dict.__contains__("measure"):
            quality_element.measure = json_dict["measure"]
        if json_dict.__contains__("evaluationMethod"):
            quality_element.evaluation_method = json_dict["evaluationMethod"]
        if json_dict.__contains__("result"):
            quality_element.result = json_dict["resule"]
        return quality_element


@dataclass
class DataQuality:
    """
    From ISO 19157-1 DataQuality
    """
    scope: Scope
    report: List[QualityElement] = field(default_factory=list)

    def to_dict(self):
        return {
            "scope": self.scope.to_dict() if self.scope is not None else None,
            "report": [r.to_dict() for r in self.report],
        }

    @staticmethod
    def from_dict(json_dict: dict):
        if json_dict.__contains__("scope"):
            data_quality = DataQuality(scope=Scope.from_dict(json_dict["scope"]))
            if json_dict.__contains__("report"):
                data_quality.report = [QualityElement.from_dict(element) for element in json_dict["report"]]
            return data_quality
        else:
            raise ValueError("The given json_dict is not a valid DataQuality")


@dataclass
class Label:
    """
    Basic label type
    """
    is_negative: bool = field(default=None)
    confidence: float = field(default=None)

    def to_dict(self):
        return {
            "type": "Label",
            "isNegative": self.is_negative,
            "confidence": self.confidence
        }

    @staticmethod
    def from_dict(json_dict: dict):
        if json_dict["type"] == "Label":
            label = Label()
            if json_dict.__contains__("isNegative"):
                label.is_negative = json_dict["isNegative"]
            if json_dict.__contains__("confidence"):
                label.confidence = json_dict["confidence"]
            return label
        else:
            raise ValueError("The given json_dict is not a valid Label")


@dataclass
class TrainingData:
    """
    Basic training data type
    """
    id: str
    dataset_id: str = field(default=None)
    training_type: str = field(default=None)
    number_of_labels: int = field(default=None)
    data_sources: List[str] = field(default_factory=list)
    quality: DataQuality = field(default=None)
    labeling: List[Labeling] = field(default_factory=list)
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
            "numberOfLabels": self.number_of_labels,
            "dataSources": self.data_sources,
            "quality": self.quality.to_dict() if self.quality is not None else None,
            "labeling": [ll.to_dict() for ll in self.labeling],
            "labels": [label.to_dict() for label in self.labels]
        }

    @staticmethod
    def from_dict(json_dict: dict):
        if json_dict.__contains__("id") and json_dict["type"] == "TrainingData":
            training_data = TrainingData(json_dict["id"])
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
                training_data.labels = [Label.from_dict(label) for label in json_dict["labels"]]
            return training_data
        else:
            raise ValueError("The given json_dict is not a valid TrainingData")


@dataclass
class Changeset:
    """
    Training Data Changeset
    """
    id: str
    change_count: int
    dataset_id: str = field(default=None)
    version: str = field(default=None)
    created_time: str = field(default=None)
    add: List[TrainingData] = field(default_factory=list)
    modify: List[TrainingData] = field(default_factory=list)
    delete: List[TrainingData] = field(default_factory=list)

    def to_dict(self):
        return {
            "type": "TDChangeset",
            "id": self.id,
            "changeCount": self.change_count,
            "datasetId": self.dataset_id,
            "version": self.version,
            "createdTime": self.created_time,
            "add": [td.to_dict() for td in self.add],
            "modify": [td.to_dict() for td in self.modify],
            "delete": [td.to_dict() for td in self.delete],
        }

    @staticmethod
    def from_dict(json_dict: dict):
        if json_dict.__contains__("id") and json_dict.__contains__("changeCount") \
                and json_dict["type"] == "TDChangeset":
            changeset = Changeset(json_dict["id"], json_dict["changeCount"])
            if json_dict.__contains__("datasetId"):
                changeset.dataset_id = json_dict["datasetId"]
            if json_dict.__contains__("version"):
                changeset.version = json_dict["version"]
            if json_dict.__contains__("createdTime"):
                changeset.created_time = json_dict["createdTime"]
            if json_dict.__contains__("add"):
                changeset.add = [TrainingData.from_dict(td) for td in json_dict["add"]]
            if json_dict.__contains__("modify"):
                changeset.add = [TrainingData.from_dict(td) for td in json_dict["modify"]]
            if json_dict.__contains__("delete"):
                changeset.add = [TrainingData.from_dict(td) for td in json_dict["delete"]]
            return changeset
        else:
            raise ValueError("The given json_dict is not a valid Changeset")


@dataclass
class TrainingDataset:
    """
    Basic training dataset type
    """
    id: str
    name: str
    description: str
    license: str
    doi: str = field(default=None)
    scope: Scope = field(default=None)
    version: str = field(default=None)
    amount_of_training_data: int = field(default=None)
    created_time: str = field(default=None)
    updated_time: str = field(default=None)
    providers: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    metrics_in_literature: List[MetricsInLiterature] = field(default_factory=list)
    statistics_info: List[KeyValuePair] = field(default_factory=list)
    dataSources: List[str] = field(default_factory=list)
    number_of_classes: int = field(default=None)
    classification_schema: str = field(default=None)
    classes: Union[List[KeyValuePair], List[str]] = field(default=None)
    tasks: List[Task] = field(default_factory=list)
    labeling: List[Labeling] = field(default_factory=list)
    quality: DataQuality = field(default=None)
    changesets: List[Changeset] = field(default_factory=list)
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
        Return the labeling of the training dataset
        """
        return self.labeling

    def get_tasks(self) -> List[Task]:
        """
        Return the tasks of the training dataset
        """
        return self.tasks

    def get_quality(self) -> DataQuality:
        """
        Return the quality of the training dataset
        """
        return self.quality

    def get_changesets(self) -> List[Changeset]:
        """
        Return the changesets of the training dataset
        """
        return self.changesets

    def to_dict(self):
        return {
            "type": "TrainingDataset",
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
            "labeling": [ll.to_dict() for ll in self.labeling],
            "quality": self.quality.to_dict() if self.quality is not None else None,
            "changesets": [changeset.to_dict() for changeset in self.changesets],
            "data": [d.to_dict() for d in self.data]
        }

    @staticmethod
    def from_dict(json_dict: dict):
        if json_dict.__contains__("id") and json_dict.__contains__("name") \
                and json_dict.__contains__("description") and json_dict.__contains__("license") \
                and json_dict["type"] == "TrainingDataset":
            td = TrainingDataset(json_dict["id"], json_dict["name"], json_dict["description"], json_dict["license"])
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
                td.tasks = [Task.from_dict(t) for t in json_dict["tasks"]]
            if json_dict.__contains__("labeling"):
                td.labeling = [Labeling.from_dict(labeling) for labeling in json_dict["labeling"]]
            if json_dict.__contains__("quality"):
                td.quality = DataQuality.from_dict(json_dict["quality"])
            if json_dict.__contains__("changesets"):
                td.changesets = [Changeset.from_dict(changeset) for changeset in json_dict["changesets"]]
            if json_dict.__contains__("data"):
                td.data = [TrainingData.from_dict(data) for data in json_dict["data"]]
            return td
        else:
            raise ValueError("The given json_dict is not a valid TrainingDataset")
