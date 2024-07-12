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

from typing import List, Union, Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from pytdml.type._utils import _validate_date, to_camel, _valid_methods, _validate_training_type


class BaseCamelModel(BaseModel):
    """
    Basic model with camel case alias
    Since python use snake case as default
    We need to convert it to camel case for JSON
    """

    class Config:
        alias_generator = to_camel
        populate_by_name = True
        arbitrary_types_allowed = True

    def json(self, **kwargs):
        return super().json(by_alias=True, **kwargs)


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
    extent: Optional[List[float]] = Field(min_items=4)
    level_description: Optional[MD_ScopeDescription]


class CI_Date(BaseCamelModel):
    """
    From ISO 19115-1 CI_Date
    """

    date: str
    dateType: str

    @field_validator("date")
    def validate_date(cls, v):
        return _validate_date(v)


class MD_BrowseGraphic(BaseCamelModel):
    """
    From ISO 19115-1 MD_BrowseGraphic
    """

    file_name: str
    file_description: Optional[str]
    file_type: Optional[str]


class CI_Citation(BaseCamelModel):
    """
    From ISO 19115-1 CI_Citation
    """

    title: str
    alternateTitle: Optional[List[str]]
    date: Optional[CI_Date]
    edition: Optional[str]
    edition_date: Optional[str]
    graphic: Optional[List[MD_BrowseGraphic]]
    identifier: Optional[List[KeyValuePair]]
    ISBN: Optional[str]
    ISSN: Optional[str]

    @field_validator("edition_date")
    def validate_edition_date(cls, v):
        return _validate_date(v)


class LinearRing(BaseCamelModel):
    """
    gml: LinearRing - NIEM 2.1
    """

    posList: List[int] = Field(min_items=4)


class LinearRing_Object(BaseCamelModel):
    """
    LinearRing Object Type
    """

    linearRing: LinearRing = Field(alias="LinearRing")


class Polygon(BaseCamelModel):
    """
    gml: Polygon - NIEM 2.1
    """

    description: Optional[str]
    description_Reference: Optional[str]
    identifier: Optional[str]
    name: Optional[List[str]]
    exterior: Optional[LinearRing_Object]
    interior: Optional[List[LinearRing_Object]]


class MD_Identifier(BaseCamelModel):
    """
    From ISO 19115-1 MD_Identifier
    """

    code: str
    authority: Optional[CI_Citation]
    code_space: Optional[str]
    version: Optional[str]
    description: Optional[str]


class BoundingPolygon(BaseCamelModel):
    """
    From ISO 19115:2003
    """

    polygon: List[Polygon] = Field(min_items=1)
    extentTypeCode: Optional[bool]


class GeographicBoundingBox(BaseCamelModel):
    """
    From ISO 19168-2:2022
    """

    westBoundLongitude: int
    eastBoundLongitude: int
    southBoundLatitude: int
    northBoundLatitude: int

    extentTypeCode: Optional[bool]


class GeographicDescription(BaseCamelModel):
    """
    From ISO 19115: 2003 Metadata
    """
    geographicIdentifier: MD_Identifier

    extentTypeCode: Optional[bool]


class TimeInstant(BaseCamelModel):
    """
    Time Instant Type
    """

    timePosition: str

    description: Optional[str]
    descriptionReference: Optional[str]
    identifier: Optional[str]
    name: Optional[List[str]]
    relatedTime: Optional[List[KeyValuePair]]


class TimePeriod(BaseCamelModel):
    """
    Time Period Type
    """

    beginPosition: str
    endPosition: str

    description: Optional[str]
    descriptionReference: Optional[str]
    identifier: Optional[str]
    name: Optional[List[str]]
    duration: Optional[str]
    timeInterval: Optional[int]
    relatedTime: Optional[List[KeyValuePair]]

    @field_validator("beginPosition")
    def validate_date_time(cls, v):
        return _validate_date(v)

    @field_validator("endPosition")
    def validate_date_time(cls, v):
        return _validate_date(v)


class TemporalExtent(BaseCamelModel):
    """
    From ISO 19108:2002
    """

    extent: Union[TimeInstant, TimePeriod]


class ReferenceSystem(BaseCamelModel):
    """
    From ISO 19111: 2019
    """

    referenceSystemIdentifier: Optional[MD_Identifier]
    referenceSystemType: Optional[str]


class VerticalCRS(BaseCamelModel):
    """
    From ISO 19111 edition 2
    """

    identifier: str
    scope: List[str] = Field(min_items=1)
    verticalCS: List[str] = Field(min_items=1)
    verticalDatum: List[str] = Field(min_items=1)

    description: Optional[str]
    description_Reference: Optional[str]
    name: Optional[List[str]]
    remarks: Optional[List[str]]
    domain_Of_Validity: Optional[List[str]]


class VerticalExtent(BaseCamelModel):
    """
    From ISO 19115 SpiralTracker Report
    """

    minimumValue: int
    maximumValue: int

    verticalCRSId: Optional[ReferenceSystem]
    verticalCRS: Optional[VerticalCRS]


class SpatialTemporalExtent(BaseCamelModel):
    """
    From ISO 19115: 2003 Metadata
    """

    extent: Union[TimeInstant, TimePeriod]
    spatialExtent: Union[BoundingPolygon, GeographicBoundingBox, GeographicDescription]

    verticalExtent: Optional[VerticalExtent]


class Extent(BaseCamelModel):
    """
    From ISO 19115: 2003 Metadata
    """

    description: str
    geographicElement: List[Union[BoundingPolygon, GeographicBoundingBox, GeographicDescription]]
    temporalElement: List[Union[TemporalExtent, SpatialTemporalExtent]]
    verticalElement: List[VerticalExtent]


class BoundingBox(BaseCamelModel):
    """
    From GeoJSON bounding box
    """

    extent: List[int] = Field(min_items=4)


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
    type: Literal["AI_AbstractTask"]

    dataset_id: Optional[str]
    description: Optional[str] = ""


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
    methods: List[str] = Field(min_items=1)

    tools: Optional[List[str]]

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

    type: Literal["AI_AbstractLabel"]

    is_negative: Optional[bool] = False  # Optional without default value
    confidence: Optional[float] = Field(ge=0.0, le=1.0)  # Optional without default value


class TrainingData(BaseCamelModel):
    """
    Basic training data type
    """

    type: Literal["AI_AbstractTrainingData"]
    id: str
    labels: List[Union[Label, "PixelLabel", "ObjectLabel", "SceneLabel"]]

    dataset_id: Optional[str]
    data_sources: Optional[List[CI_Citation]] = None
    number_of_labels: Optional[int]
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

    add: Optional[List[TrainingData]]
    change_count: Optional[int]
    dataset_id: Optional[str]
    delete: Optional[List[TrainingData]]
    modify: Optional[List[TrainingData]]
    version: Optional[str]
    created_time: Optional[str]

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

    type: Optional[List[StatisticsInfoType]] = Field(min_items=1)


class AI_TDChangeset(BaseCamelModel):
    type: Literal["AI_TDChangeset"]
    id: str
    change_count: int

    dataset_id: Optional[str]
    version: Optional[str]
    created_time: Optional[str]
    add: Optional[List[Union[TrainingData, "EOTrainingData"]]]
    modify: Optional[List[Union[TrainingData, "EOTrainingData"]]]
    delete: Optional[List[Union[TrainingData, "EOTrainingData"]]]

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
    tasks: List[Union[Task, "EOTask"]] = Field(min_items=1)
    data: List[Union[TrainingData, "EOTrainingData"]] = Field(min_items=1)  # That one should be uri-format
    type: Literal["AI_AbstractTrainingDataset"]

    amount_Of_TrainingData: Optional[int]
    classes: Optional[List[NamedValue]]
    classification_schema: Optional[str] = ""  # That one should be uri-format
    created_time: Optional[str]
    dataSources: Optional[List[CI_Citation]] = []  # That string one should be uri-format
    doi: Optional[str] = ""
    keywords: Optional[List[str]]
    number_of_classes: Optional[int]
    providers: Optional[List[str]] = []
    scope: Optional[MD_Scope] = None
    statistics_info: Optional[List[NamedValue]] = None
    updated_time: Optional[str] = ""
    version: Optional[str]
    labeling: Optional[List[Labeling]] = []
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
