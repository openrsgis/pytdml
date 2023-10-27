# ------------------------------------------------------------------------------
#
# Project: pytdml
# Authors: Boyi Shangguan, Kaixuan Wang, Zhaoyan Wu
# Created: 2022-05-04
# Modified: 2023-10-27
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

from typing import List, Union, Optional, Dict, Literal
from pydantic import BaseModel, Field, validator, root_validator
from datetime import datetime
from pytdml.type._utils import _validate_date, to_camel


class BaseCamelModel(BaseModel):
    """
    Basic model with camel case alias
    Since python use snake case as default
    We need to convert it to camel case for JSON
    """
    class Config:
        alias_generator = to_camel
        allow_population_by_field_name = True

    def json(self, **kwargs):
        return super().json(by_alias=True, **kwargs)


class KeyValuePair(BaseCamelModel):
    """
    Key/Value pair type
    """

    key: str
    value: Union[str, int, float, bool, None]


class MD_ScopeDescription(BaseCamelModel):
    """
    From ISO 19115-1 MD_ScopeDescription
    """

    dataset: str
    features: Optional[List[str]]


class MD_Band(BaseCamelModel):
    """
    From ISO 19115-1 MD_Band
    """
    bound_max: Optional[float]
    bound_min: Optional[float]
    bound_units: Optional[Literal["nm", "um", "cm", "dm", "m", "km"]]
    peak_response: Optional[float]
    tone_gradation: Optional[int]

    @root_validator(pre=True)
    def check_dependent_required(cls, v):
        bound_units = v.get("boundUnits")
        bound_max = v.get("boundMax")
        bound_min = v.get("boundMin")

        # check dependRequired in jsonschema
        if bound_units is not None and (bound_max is None or bound_min is None):
            raise ValueError(
                "boundMax and boundMin are required when boundUnits is present"
            )

        return v


class MD_Scope(BaseCamelModel):
    """
    From ISO 19115-1 MD_Scope
    """

    level: str
    extent: Optional[List[float]] = Field(min_items=4)
    level_description: Optional[MD_ScopeDescription]


class CIDate(BaseCamelModel):
    """
    From ISO 19115-1 CI_Date
    """

    date: str
    dateType: str

    @validator("date")
    def validate_date(cls, v):
        return _validate_date(v)


class MD_BrowseGraphic(BaseCamelModel):
    """From ISO 19115-1 MD_BrowseGraphic"""

    file_name: str
    file_description: Optional[str]
    file_type: Optional[str]


class CICitation(BaseCamelModel):
    """
    From ISO 19115-1 CI_Citation
    """

    title: str
    alternateTitle: Optional[List[str]]
    date: Optional[CIDate]
    edition: Optional[str]
    edition_date: Optional[str]
    graphic: Optional[List[MD_BrowseGraphic]]
    identifier: Optional[List[KeyValuePair]]
    ISBN: Optional[str]
    ISSN: Optional[str]

    @validator("edition_date")
    def validate_edition_date(cls, v):
        return _validate_date(v)


class MD_Identifier(BaseCamelModel):
    """
    From ISO 19115-1 MD_Identifier
    """

    code: str
    authority: Optional[CICitation]
    code_space: Optional[str]
    version: Optional[str]
    description: Optional[str]


class MetricsPair(BaseCamelModel):
    """
    Metrics pair type
    """

    name: str
    value: float


class MetricsInLiterature(BaseCamelModel):
    """
    Metrics in literature type
    """

    doi: str
    algorithm: str = None
    metrics: Optional[List[MetricsPair]] = Field(min_items=1)


class Task(BaseCamelModel):
    """
    Basic task type
    """

    id: str
    dataset_id: Optional[str]
    description: Optional[str]


class Labeler(BaseCamelModel):
    """
    Labeler type
    """

    id: str
    name: str
    type: Literal["AI_Labeler"]


class LabelingProcedure(BaseCamelModel):
    """
    Labeling procedure type
    """

    type: Literal["AI_LabelingProcedure"]
    id: str
    methods: List[str] = []
    tools: Optional[List[str]]


class Labeling(BaseCamelModel):
    """
    Labeling type
    """

    id: str
    scope: MD_Scope
    type: Literal["AI_Labeling"]
    labelers: Optional[List[Labeler]]
    procedure: Optional[LabelingProcedure]


class QualityElement(BaseCamelModel):
    """
    From ISO 19157-1 QualityElement
    """

    type: str
    measure: str
    evaluation_method: str
    result: List[str]


class DataQuality(BaseCamelModel):
    """
    From ISO 19157-1 DataQuality
    """

    type: Literal["DataQuality"]
    scope: MD_Scope
    report: Optional[List[QualityElement]]


class Label(BaseCamelModel):
    """
    Basic label type
    """

    is_negative: Optional[bool]  # Optional without default value
    confidence: Optional[float] = Field(
        ge=0.0, le=1.0
    )  # Optional without default value


class TrainingData(BaseCamelModel):
    """
    Basic training data type
    """

    id: str
    labels: List[Label]
    dataset_id: Optional[str]
    training_type: Optional[str]
    number_of_labels: Optional[int]
    data_sources: Optional[Union[List[str], List[CICitation]]]
    quality: Optional[DataQuality]
    labeling: Optional[List[Labeling]]
    quality: Optional[DataQuality]


class Changeset(BaseCamelModel):
    """
    Training Data Changeset
    """

    type: Literal["AI_TDChangeset"]
    id: str
    change_count: int

    add: Optional[List[TrainingData]]
    change_count: Optional[int]
    dataset_id: Optional[str]
    delete: Optional[List[TrainingData]]
    modify: Optional[List[TrainingData]]
    version: Optional[str]
    created_time: Optional[str]

    @validator("created_time")
    def validate_created_time(cls, v):
        return _validate_date(v)


class StatisticsInfoType(BaseCamelModel):
    """
    Statistics info type
    """

    key: str
    value: int


class StatisticsInfo(BaseCamelModel):
    """
    Statistics info
    """

    type: Optional[List[StatisticsInfoType]] = Field(min_items=1)


class TrainingDataset(BaseCamelModel):
    """
    Basic training dataset type
    """

    id: str
    name: str
    description: str
    license: str
    tasks: List[Task]
    data: Union[List[TrainingData], str]  # That one should be uri-format

    amount_of_training_data: Optional[int]
    classes: Optional[List[Union[KeyValuePair, str, Dict]]] = Field(min_items=1)
    classification_schema: Optional[str]  # That one should be uri-format
    created_time: Optional[str]
    dataSources: Optional[
        Union[List[str], List[CICitation]]
    ]  # That string one should be uri-format
    doi: Optional[str]
    keywords: Optional[List[str]]
    number_of_classes: Optional[int]
    scope: Optional[MD_Scope]
    version: Optional[str]
    updated_time: Optional[str]
    labeling: Optional[List[Labeling]]
    metrics_in_LIT: Optional[List[MetricsInLiterature]]
    quality: Optional[DataQuality]
    providers: Optional[List[str]]
    statistics_info: Optional[StatisticsInfo]

    @validator("created_time")
    def validate_created_time(cls, v):
        return _validate_date(v)

    @validator("updated_time")
    def validate_updated_time(cls, v):
        return _validate_date(v)
