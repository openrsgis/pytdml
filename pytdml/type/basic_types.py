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
import json
from typing import List, Union, Optional, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from pytdml.type._utils import _validate_date, to_camel, _valid_methods, _validate_training_type


class BaseCamelModel(BaseModel):
    """
    Basic model with camel case alias
    Since python use snake case as default
    We need to convert it to camel case for JSON
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    def model_dump_json(self, **kwargs):
        return super().model_dump_json(by_alias=True, **kwargs)
    
    def model_dump(self, **kwargs):
        if "exclude_none" not in kwargs:
            kwargs["exclude_none"] = True
        return super().model_dump(**kwargs)

    @classmethod
    def read_json(cls, file_path: str):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return cls(**data)

    @classmethod
    def read_yaml(cls, file_path: str):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return cls(**data)

    def write_json(self, file_path: str) -> None:
        with open(file_path, 'w') as file:
            json.dump(self.model_dump(), file)

    def write_yaml(self, file_path: str) -> None:
        with open(file_path, 'w') as file:
            yaml.dump(self.model_dump(), file, default_flow_style=False)


class KeyValuePair(BaseCamelModel):
    """
    Key/Value pair type
    """

    key: str
    value: Union[str, int, float, bool, None]


class NamedValue(BaseCamelModel):
    """
    From ISO 19156: 2023(E) NamedValue
    """

    key: str
    value: Union[str, object, int, list, bool, None]


class MD_ScopeDescription(BaseCamelModel):
    """
    From ISO 19115-1 MD_ScopeDescription
    """

    dataset: str
    features: Optional[List[str]] = None


class MD_Band(BaseCamelModel):
    """
    From ISO 19115-1 MD_Band
    """

    name: Optional[List["MD_Identifier"]] = Field(default=None)
    bound_max: Optional[float] = None
    bound_min: Optional[float] = None
    bound_units: Optional[Literal["nm", "um", "cm", "dm", "m", "km"]] = None
    peak_response: Optional[float] = None
    tone_gradation: Optional[int] = None

    @model_validator(mode="before")
    def check_dependent_required(self):
        bound_units = self.get("boundUnits")
        bound_max = self.get("boundMax")
        bound_min = self.get("boundMin")

        # check dependRequired in jsonschema
        if bound_units is not None and (bound_max is None or bound_min is None):
            raise ValueError(
                "boundMax and boundMin are required when boundUnits is present"
            )

        return self


class MD_Scope(BaseCamelModel):
    """
    From ISO 19115-1 MD_Scope
    """

    level: str
    extent: Optional[List[float]] = Field(min_length=4, default=None)
    level_description: Optional[MD_ScopeDescription] = None


class CI_Date(BaseCamelModel):
    """
    From ISO 19115-1 CI_Date
    """

    date: str
    date_type: str

    @field_validator("date")
    def validate_date(cls, v):
        return _validate_date(v)


class MD_BrowseGraphic(BaseCamelModel):
    """
    From ISO 19115-1 MD_BrowseGraphic
    """

    file_name: str
    file_description: Optional[str] = None
    file_type: Optional[str] = None


class CI_Citation(BaseCamelModel):
    """
    From ISO 19115-1 CI_Citation
    """

    title: str
    alternate_title: Optional[List[str]] = None
    date: Optional[CI_Date] = None
    edition: Optional[str] = None
    edition_date: Optional[str] = None
    graphic: Optional[List[MD_BrowseGraphic]] = None
    identifier: Optional[List[KeyValuePair]] = None
    ISBN: Optional[str] = None
    ISSN: Optional[str] = None

    @field_validator("edition_date")
    def validate_edition_date(cls, v):
        return _validate_date(v)


class LinearRing(BaseCamelModel):
    """
    gml: LinearRing - NIEM 2.1
    """

    posList: List[int] = Field(min_length=4)


class LinearRing_Object(BaseCamelModel):
    """
    LinearRing Object Type
    """

    linear_ring: LinearRing = Field(alias="LinearRing")  # Why are all the rest camel case and this one is not?


class Polygon(BaseCamelModel):
    """
    gml: Polygon - NIEM 2.1
    """

    description: Optional[str] = None
    description_reference: Optional[str] = None
    identifier: Optional[str] = None
    name: Optional[List[str]] = None
    exterior: Optional[LinearRing_Object] = None
    interior: Optional[List[LinearRing_Object]] = None


class MD_Identifier(BaseCamelModel):
    """
    From ISO 19115-1 MD_Identifier
    """

    code: str
    authority: Optional[CI_Citation] = None
    code_space: Optional[str] = None
    version: Optional[str] = None
    description: Optional[str] = None


class BoundingPolygon(BaseCamelModel):
    """
    From ISO 19115:2003
    """

    polygon: List[Polygon] = Field(min_length=1)
    extent_type_code: Optional[bool] = None


class GeographicBoundingBox(BaseCamelModel):
    """
    From ISO 19168-2:2022
    """

    west_bound_longitude: int
    east_bound_longitude: int
    south_bound_latitude: int
    north_bound_latitude: int

    extent_type_code: Optional[bool] = None


class GeographicDescription(BaseCamelModel):
    """
    From ISO 19115: 2003 Metadata
    """
    geographic_identifier: MD_Identifier
    extent_type_code: Optional[bool] = None


class TimeInstant(BaseCamelModel):
    """
    Time Instant Type
    """

    timePosition: str

    description: Optional[str] = None
    description_reference: Optional[str] = None
    identifier: Optional[str] = None
    name: Optional[List[str]] = None
    related_time: Optional[List[KeyValuePair]] = None


class TimePeriod(BaseCamelModel):
    """
    Time Period Type
    """

    begin_position: str
    end_position: str

    description: Optional[str] = None
    description_reference: Optional[str] = None
    identifier: Optional[str] = None
    name: Optional[List[str]] = None
    duration: Optional[str] = None
    time_interval: Optional[int] = None
    related_time: Optional[List[KeyValuePair]] = None

    @field_validator("begin_position")
    def validate_begin_date_time(cls, v):
        return _validate_date(v)

    @field_validator("end_position")
    def validate_end_date_time(cls, v):
        return _validate_date(v)


class TemporalExtent(BaseCamelModel):
    """
    From ISO 19108:2002
    """

    extent: Union[TimeInstant, TimePeriod]


class ReferenceSystem(BaseCamelModel):  # TODO: can both be None?
    """
    From ISO 19111: 2019
    """

    reference_system_identifier: Optional[MD_Identifier] = None
    reference_system_type: Optional[str] = None


class VerticalCRS(BaseCamelModel):
    """
    From ISO 19111 edition 2
    """

    identifier: str
    scope: List[str] = Field(min_length=1)
    verticalCS: List[str] = Field(min_length=1)
    vertical_datum: List[str] = Field(min_length=1)

    description: Optional[str] = None
    description_Reference: Optional[str] = None
    name: Optional[List[str]] = None
    remarks: Optional[List[str]] = None
    domain_Of_Validity: Optional[List[str]] = None  # Is this casing intentional?


class VerticalExtent(BaseCamelModel):
    """
    From ISO 19115 SpiralTracker Report
    """

    minimum_value: int
    maximum_value: int

    verticalCRSId: Optional[ReferenceSystem] = None
    verticalCRS: Optional[VerticalCRS] = None


class SpatialTemporalExtent(BaseCamelModel):
    """
    From ISO 19115: 2003 Metadata
    """

    extent: Union[TimeInstant, TimePeriod]
    spatial_extent: Union[BoundingPolygon, GeographicBoundingBox, GeographicDescription]

    vertical_extent: Optional[VerticalExtent] = None


class Extent(BaseCamelModel):
    """
    From ISO 19115: 2003 Metadata
    """

    description: str
    geographic_element: List[Union[BoundingPolygon, GeographicBoundingBox, GeographicDescription]]
    temporal_element: List[Union[TemporalExtent, SpatialTemporalExtent]]
    vertical_element: List[VerticalExtent]


class BoundingBox(BaseCamelModel):
    """
    From GeoJSON bounding box
    """

    extent: List[Union[int, float]] = Field(min_length=4)


class MetricsPair(BaseCamelModel):
    """
    Metrics pair type
    """

    name: str
    value: float


class MetricsInLiterature(BaseCamelModel):
    """
    Metrics in literature type
    """  # Why are the properties of this class specified in camel case?

    doi: str
    algorithm: Optional[str] = None
    metrics: Optional[List[MetricsPair]] = Field(min_length=1, default=None)


class Task(BaseCamelModel):
    """
    Basic task type
    """

    id: str
    type: Literal["AI_AbstractTask"]

    dataset_id: Optional[str] = None
    description: Optional[str] = None


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
    methods: List[str] = Field(min_length=1)

    tools: Optional[List[str]] = None

    @field_validator("methods")
    def valid_methods(cls, v):
        return _valid_methods(v)


class Labeling(BaseCamelModel):
    """
    Labeling type
    """

    id: str
    scope: MD_Scope
    type: Literal["AI_Labeling"]
    labelers: Optional[List[Labeler]] = None
    procedure: Optional[LabelingProcedure] = None


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
    report: Optional[List[QualityElement]] = None


class Label(BaseCamelModel):
    """
    Basic label type
    """

    type: Literal["AI_AbstractLabel"]

    is_negative: Optional[bool] = None  # Optional without default value
    confidence: Optional[float] = Field(ge=0.0, le=1.0, default=None)  # Optional without default value


class TrainingData(BaseCamelModel):
    """
    Basic training data type
    """

    type: Literal["AI_AbstractTrainingData"]
    id: str
    labels: List[Union[Label, "AI_PixelLabel", "ObjectLabel", "SceneLabel"]] = None

    dataset_id: Optional[str] = None
    data_sources: Optional[List[CI_Citation]] = None
    number_of_labels: Optional[int] = None
    labeling: Optional[List[Labeling]] = None
    training_type: Optional[str] = None
    quality: Optional[List[DataQuality]] = None

    @field_validator("training_type")
    def validate_training_type(cls, v):
        valid_format = []
        for item in v:
            valid_format.append(_validate_training_type(item))
        return valid_format


class Changeset(BaseCamelModel):
    """
    Training Data Changeset
    """

    type: Literal["AI_TDChangeset"]
    id: str
    change_count: int

    add: Optional[List[TrainingData]] = None
    change_count: Optional[int] = None
    dataset_id: Optional[str] = None
    delete: Optional[List[TrainingData]] = None
    modify: Optional[List[TrainingData]] = None
    version: Optional[str] = None
    created_time: Optional[str] = None

    @field_validator("created_time")
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

    type: Optional[List[StatisticsInfoType]] = None


class AI_TDChangeset(BaseCamelModel):
    type: Literal["AI_TDChangeset"]
    id: str
    change_count: int

    dataset_id: Optional[str] = None
    version: Optional[str] = None
    created_time: Optional[str] = None
    add: Optional[List[Union[TrainingData, "EOTrainingData"]]] = None
    modify: Optional[List[Union[TrainingData, "EOTrainingData"]]] = None
    delete: Optional[List[Union[TrainingData, "EOTrainingData"]]] = None

    @field_validator("created_time")
    def validate_created_time(cls, v):
        return _validate_date(v)


class TrainingDataset(BaseCamelModel):
    """
    Basic training dataset type
    """

    id: str
    name: str
    description: str
    license: str
    tasks: List[Union[Task, "EOTask"]] = Field(min_length=1)
    data: List[Union[TrainingData, "EOTrainingData"]] = Field(min_length=1)  # That one should be uri-format
    type: Literal["AI_AbstractTrainingDataset"]

    amount_of_training_data: Optional[int] = None
    classes: Optional[List[NamedValue]] = None
    classificationSchema: Optional[str] = None
    created_time: Optional[str] = None
    data_sources: Optional[List[CI_Citation]] = None  # That string one should be uri-format
    doi: Optional[str] = None
    keywords: Optional[List[str]] = None
    number_of_classes: Optional[int] = None
    providers: Optional[List[str]] = None
    scope: Optional[MD_Scope] = None
    statistics_info: Optional[List[NamedValue]] = None
    updated_time: Optional[str] = None
    version: Optional[str] = None
    labeling: Optional[List[Labeling]] = None
    metrics_in_LIT: Optional[List[MetricsInLiterature]] = None
    quality: Optional[List[DataQuality]] = None
    changesets: Optional[List[AI_TDChangeset]] = None

    @field_validator("created_time")
    def validate_created_time(cls, v):
        return _validate_date(v)

    @field_validator("updated_time")
    def validate_updated_time(cls, v):
        if v is not None:
            return _validate_date(v)
        else:
            return v
