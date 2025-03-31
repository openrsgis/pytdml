# ------------------------------------------------------------------------------
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

import copy
from typing_extensions import TypedDict
from typing import List, Union, Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from pytdml.type._utils import (
    _validate_date,
    to_camel,
    _valid_methods,
    _validate_training_type,
    _validate_evaluation_method_type,
)


class BaseCamelModel(BaseModel):
    """
    Basic model with camel case alias
    Since python use snake case as default
    We need to convert it to camel case for JSON
    """

    model_config: TypedDict = {
        "alias_generator": to_camel,
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
    }


class KeyValuePair(BaseCamelModel):
    """
    Key/Value pair type
    """

    key: list
    value: list

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return KeyValuePair(**new_dict)


class NamedValue(BaseCamelModel):
    """
    From ISO 19156: 2023(E) NamedValue
    """

    key: str
    value: Union[str, object, int, float, list, bool, None]

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return NamedValue(**new_dict)


class CI_Date(BaseCamelModel):
    """
    From ISO 19115-1 CI_Date
    """

    date: str
    date_type: str

    @field_validator("date")
    def validate_date(cls, v):
        return _validate_date(v)

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return CI_Date(**new_dict)


class CI_Citation(BaseCamelModel):
    """
    From ISO 19115-1 CI_Citation
    """

    title: str
    alternate_title: Optional[List[str]] = None
    date: Optional[List[CI_Date]] = None
    edition: Optional[str] = None
    edition_date: Optional[str] = None
    graphic: Optional[List[KeyValuePair]] = None
    identifier: Optional[List[KeyValuePair]] = None
    ISBN: Optional[str] = None
    ISSN: Optional[str] = None

    @field_validator("edition_date")
    def validate_edition_date(cls, v):
        return _validate_date(v)

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return CI_Citation(**new_dict)


class LinearRing(BaseCamelModel):
    """
    gml: LinearRing - NIEM 2.1
    """

    pos_list: List[float] = Field(min_length=4)

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return LinearRing(**new_dict)


class LinearRing_Object(BaseCamelModel):
    """
    LinearRing Object Type
    """

    linear_ring: LinearRing

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return LinearRing_Object(**new_dict)


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

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return Polygon(**new_dict)


class MD_Identifier(BaseCamelModel):
    """
    From ISO 19115-1 MD_Identifier
    """

    code: str
    authority: Optional[CI_Citation] = None
    code_space: Optional[str] = None
    version: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return MD_Identifier(**new_dict)


class MemberName(BaseCamelModel):
    """
    From ISO 19115-1 MD_Identifier
    """

    a_name: str
    attribute_type: str

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return MemberName(**new_dict)


class MI_RangeElementDescription(BaseCamelModel):
    """
    From ISO 19115-1 MD_Identifier
    """

    name: str
    definition: str
    range_element: List[str] = Field(min_length=1)

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return MI_RangeElementDescription(**new_dict)


class MD_Band(BaseCamelModel):
    """
    From ISO 19115-1 MD_Band
    """

    sequence_identifier: Optional[MemberName] = None
    description: Optional[str] = None
    name: Optional[List[MD_Identifier]] = None
    max_value: Optional[float] = None
    min_value: Optional[float] = None
    units: Optional[str] = None
    scale_factor: Optional[float] = None
    offset: Optional[float] = None
    mean_value: Optional[float] = None
    number_of_values: Optional[int] = None
    standard_deviation: Optional[float] = None
    other_property_type: Optional[float] = None
    other_property: Optional[str] = None
    bits_per_value: Optional[int] = None
    range_element_description: Optional[List[MI_RangeElementDescription]] = None
    bound_max: Optional[float] = None
    bound_min: Optional[float] = None
    bound_units: Optional[str] = None
    peak_response: Optional[float] = None
    tone_gradation: Optional[int] = None

    @model_validator(mode="before")
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

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return MD_Band(**new_dict)


class EX_BoundingPolygon(BaseCamelModel):
    """
    From ISO 19115:2003
    """

    polygon: List[Polygon] = Field(min_length=1)
    extent_type_code: Optional[bool] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return EX_BoundingPolygon(**new_dict)


class EX_GeographicBoundingBox(BaseCamelModel):
    """
    From ISO 19168-2:2022
    """

    west_bound_longitude: int
    east_bound_longitude: int
    south_bound_latitude: int
    north_bound_latitude: int

    extent_type_code: Optional[bool] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return EX_GeographicBoundingBox(**new_dict)


class EX_GeographicDescription(BaseCamelModel):
    """
    From ISO 19115: 2003 Metadata
    """

    geographic_identifier: MD_Identifier

    extent_type_code: Optional[bool] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return EX_GeographicDescription(**new_dict)


class TimeInstant(BaseCamelModel):
    """
    Time Instant Type
    """

    time_position: str

    description: Optional[str] = None
    description_reference: Optional[str] = None
    identifier: Optional[str] = None
    name: Optional[List[str]] = None
    related_time: Optional[List[KeyValuePair]] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return TimeInstant(**new_dict)


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
    def validate_begin_position(cls, v):
        return _validate_date(v)

    @field_validator("end_position")
    def validate_end_position(cls, v):
        return _validate_date(v)

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return TimePeriod(**new_dict)


class EX_TemporalExtent(BaseCamelModel):
    """
    From ISO 19108:2002
    """

    extent: Union[TimeInstant, TimePeriod]

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return EX_TemporalExtent(**new_dict)


class EX_ReferenceSystem(BaseCamelModel):
    """
    From ISO 19111: 2019
    """

    reference_system_identifier: Optional[MD_Identifier] = None
    reference_system_type: Optional[str] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return EX_ReferenceSystem(**new_dict)


class VerticalCRS(BaseCamelModel):
    """
    From ISO 19111 edition 2
    """

    identifier: str
    scope: List[str] = Field(min_length=1)
    vertical_CS: List[str] = Field(min_length=1)
    vertical_datum: List[str] = Field(min_length=1)

    description: Optional[str] = None
    description_reference: Optional[str] = None
    name: Optional[List[str]] = None
    remarks: Optional[List[str]] = None
    domain_of_validity: Optional[List[str]] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return VerticalCRS(**new_dict)


class EX_VerticalExtent(BaseCamelModel):
    """
    From ISO 19115 SpiralTracker Report
    """

    minimum_value: int
    maximum_value: int

    vertical_CRS_id: Optional[EX_ReferenceSystem] = None
    vertical_CRS: Optional[VerticalCRS] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return EX_VerticalExtent(**new_dict)


class EX_SpatialTemporalExtent(BaseCamelModel):
    """
    From ISO 19115: 2003 Metadata
    """

    extent: Union[TimeInstant, TimePeriod]
    spatial_extent: Union[
        EX_BoundingPolygon, EX_GeographicBoundingBox, EX_GeographicDescription
    ]

    vertical_extent: Optional[EX_VerticalExtent] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return EX_SpatialTemporalExtent(**new_dict)


class EX_Extent(BaseCamelModel):
    """
    From ISO 19115: 2003 Metadata
    """

    description: Optional[str] = None
    geographic_element: Optional[
        List[
            Union[
                EX_BoundingPolygon, EX_GeographicBoundingBox, EX_GeographicDescription
            ]
        ]
    ] = None
    temporal_element: Optional[
        List[Union[EX_TemporalExtent, EX_SpatialTemporalExtent]]
    ] = None
    vertical_element: Optional[List[EX_VerticalExtent]] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return EX_Extent(**new_dict)


# class BoundingBox(BaseCamelModel):
#     """
#     From GeoJSON bounding box
#     """
#
#     extent: List[float] = Field(min_length=4)
#
#     def to_dict(self):
#         return self.model_dump(by_alias=True, exclude_none=True)
#
#     @staticmethod
#     def from_dict(json_dict):
#         return BoundingBox(**json_dict)


class MD_ScopeDescription(BaseCamelModel):
    """
    From ISO 19115-1 MD_ScopeDescription
    """

    attributes: Optional[str] = None
    features: Optional[str] = None
    feature_instances: Optional[str] = None
    attribute_instances: Optional[str] = None
    dataset: Optional[str] = None
    other: Optional[str] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return MD_ScopeDescription(**new_dict)


class MD_Scope(BaseCamelModel):
    """
    From ISO 19115-1 MD_Scope
    """

    level: str
    extent: Optional[List[Union[EX_Extent, List[float]]]] = None
    level_description: Optional[List[MD_ScopeDescription]] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return MD_Scope(**new_dict)


class CI_Telephone(BaseCamelModel):

    number: str
    number_type: Optional[str] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return CI_Telephone(**new_dict)


class CI_Address(BaseCamelModel):

    delivery_point: Optional[List[str]] = None
    city: Optional[str] = None
    administrative_area: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = None
    electronic_mail_address: Optional[List[str]] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return CI_Address(**new_dict)


class CI_OnlineResource(BaseCamelModel):

    linkage: str
    protocol: Optional[str] = None
    application_profile: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    function: Optional[str] = None
    protocol_request: Optional[str] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return CI_OnlineResource(**new_dict)


class CI_Contact(BaseCamelModel):

    phone: Optional[List[CI_Telephone]] = None
    address: Optional[List[CI_Address]] = None
    online_resource: Optional[List[CI_OnlineResource]] = None
    hours_of_service: Optional[List[str]] = None
    contact_instructions: Optional[str] = None
    contact_type: Optional[str] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return CI_Contact(**new_dict)


class CI_Individual(BaseCamelModel):

    name: Optional[str] = None
    contact_info: Optional[List[CI_Contact]] = None
    party_identifier: Optional[List[MD_Identifier]] = None
    position_name: Optional[str] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return CI_Individual(**new_dict)


class CI_Organisation(BaseCamelModel):

    name: Optional[str] = None
    contact_info: Optional[List[CI_Contact]] = None
    party_identifier: Optional[List[MD_Identifier]] = None
    logo: Optional[List[KeyValuePair]] = None
    individual: Optional[List[CI_Individual]] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return CI_Organisation(**new_dict)


class CI_Responsibility(BaseCamelModel):

    role: str
    party: List[Union[CI_Individual, CI_Organisation]]

    extent: Optional[List[Union[EX_Extent, List[float]]]] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return CI_Responsibility(**new_dict)


class MD_Releasability(BaseCamelModel):

    addressee: Optional[List[CI_Responsibility]] = None
    statement: Optional[str] = None
    dissemination_constraints: Optional[List[str]] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return MD_Releasability(**new_dict)


class MD_Constraints(BaseCamelModel):

    use_limitation: Optional[List[str]] = None
    constraint_application_scope: Optional[MD_Scope] = None
    graphic: Optional[List[KeyValuePair]] = None
    reference: Optional[List[CI_Citation]] = None
    releasability: Optional[MD_Releasability] = None
    responsible_party: Optional[List[CI_Responsibility]] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return MD_Constraints(**new_dict)


class MD_BrowseGraphic(BaseCamelModel):
    """
    From ISO 19115-1 MD_BrowseGraphic
    """

    file_name: str
    file_description: Optional[str] = None
    file_type: Optional[str] = None
    image_constraints: Optional[List[MD_Constraints]] = None
    linkage: Optional[List[CI_OnlineResource]] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return MD_BrowseGraphic(**new_dict)


# class MetricsPair(BaseCamelModel):
#     """
#     Metrics pair type
#     """
#
#     name: str
#     value: float
#
#     def to_dict(self):
#         return self.model_dump(by_alias=True, exclude_none=True)
#
#     @staticmethod
#     def from_dict(json_dict):
#         new_dict = copy.deepcopy(json_dict)
#         return MetricsPair(**new_dict)


class AI_MetricsInLiterature(BaseCamelModel):
    """
    Metrics in literature type
    """

    doi: str
    metrics: List[NamedValue] = Field(min_length=1)

    algorithm: Optional[str] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return AI_MetricsInLiterature(**new_dict)


class AI_Task(BaseCamelModel):
    """
    Basic task type
    """

    id: str
    type: Literal["AI_AbstractTask"]

    dataset_id: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return AI_Task(**new_dict)


class AI_Labeler(BaseCamelModel):
    """
    Labeler type
    """

    id: str
    name: str
    type: Literal["AI_Labeler"]

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return AI_Labeler(**new_dict)


class AI_LabelingProcedure(BaseCamelModel):
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

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return AI_LabelingProcedure(**new_dict)


class AI_Labeling(BaseCamelModel):
    """
    Labeling type
    """

    id: str
    scope: MD_Scope
    type: Literal["AI_Labeling"]

    labelers: Optional[List[AI_Labeler]] = None
    procedure: Optional[AI_LabelingProcedure] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return AI_Labeling(**new_dict)


class MeasureReference(BaseCamelModel):

    measure_identification: Optional[MD_Identifier] = None
    name_of_measure: Optional[List[str]] = None
    measure_description: Optional[str] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return MeasureReference(**new_dict)


class EvaluationMethod(BaseCamelModel):

    name: Optional[str] = None
    evaluation_method_description: Optional[str] = None
    evaluation_method_type: Optional[List[str]] = None
    evaluation_procedure: Optional[CI_Citation] = None
    date_time: Optional[List[str]] = None
    reference_doc: Optional[List[CI_Citation]] = None
    deductive_source: Optional[str] = None

    @field_validator("date_time")
    def validate_date_time(cls, v):
        validated_date = []
        for item in v:
            validated_date.append(_validate_date(item))
        return validated_date

    @field_validator("evaluation_method_type")
    def validate_data_time(cls, v):
        evaluation_method = []
        for item in v:
            if _validate_evaluation_method_type(item):
                evaluation_method.append(item)
        return evaluation_method

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return EvaluationMethod(**new_dict)


class QuantitativeResult(BaseCamelModel):

    value: List[Union[str, object, int, float, list, bool, None]]

    value_unit: Optional[str] = None
    value_record_type: Optional[str] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return QuantitativeResult(**new_dict)


class ConformanceResult(BaseCamelModel):

    pass_bool: bool = Field(alias="pass")
    specification: CI_Citation

    explanation: Optional[List[str]] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return ConformanceResult(**new_dict)


class DescriptiveResult(BaseCamelModel):

    statement: str

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return DescriptiveResult(**new_dict)


class MD_Dimension(BaseCamelModel):

    dimension_name: str
    dimension_size: int

    resolution: Optional[str] = None
    dimension_title: Optional[str] = None
    dimension_description: Optional[str] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return MD_Dimension(**new_dict)


class MD_GridSpatialRepresentation(BaseCamelModel):

    number_of_dimensions: int
    cell_geometry: str
    transformation_parameter_availability: bool

    scope: Optional[MD_Scope] = None
    axis_dimension_properties: Optional[List[MD_Dimension]]

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return MD_GridSpatialRepresentation(**new_dict)


class MD_GeometricObjects(BaseCamelModel):

    geometric_object_type: str
    geometric_object_count: Optional[int]

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return MD_GeometricObjects(**new_dict)


class MD_VectorSpatialRepresentation(BaseCamelModel):

    scope: Optional[MD_Scope]
    topology_level: Optional[str]
    geometric_objects: Optional[MD_GeometricObjects]

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return MD_VectorSpatialRepresentation(**new_dict)


class MD_RangeDimension(BaseCamelModel):

    sequence_identifier: Optional[MemberName] = None
    description: Optional[str] = None
    name: Optional[List[MD_Identifier]] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return MD_RangeDimension(**new_dict)


class CoverageResult(BaseCamelModel):

    spatial_representation_type: str
    result_spatial_representation: Union[
        MD_GridSpatialRepresentation, MD_VectorSpatialRepresentation
    ]

    result_content: Optional[List[MD_RangeDimension]] = None
    result_format: Optional[str] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return CoverageResult(**new_dict)


class QualityElement(BaseCamelModel):
    """
    From ISO 19157-1 QualityElement
    """

    type: str
    measure: MeasureReference
    evaluation_method: EvaluationMethod
    result: List[
        Union[QuantitativeResult, ConformanceResult, DescriptiveResult, CoverageResult]
    ]

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return QualityElement(**new_dict)


class DataQuality(BaseCamelModel):
    """
    From ISO 19157-1 DataQuality
    """

    type: Literal["DataQuality"]
    scope: MD_Scope
    report: Optional[List[QualityElement]] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return DataQuality(**new_dict)


# class GeoJSON Point
#
#
# class Feature(BaseCamelModel):
#
#     type: Literal["Feature"]
#     properties: Union[None, object]
#     geometry: Union[]
#
#     id: Optional[Union[str, float]] = None
#     bbox: Optional[List[float]] = Field(min_length=4)


class AI_Label(BaseCamelModel):
    """
    Basic label type
    """

    type: Literal["AI_AbstractLabel"]

    is_negative: Optional[bool] = Field(False)
    confidence: Optional[float] = Field(1.0, ge=0.0, le=1.0)

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return AI_Label(**new_dict)


class AI_TrainingData(BaseCamelModel):
    """
    Basic training data type
    """

    type: Literal["AI_AbstractTrainingData"]
    id: str
    labels: List[Union[AI_Label, "AI_PixelLabel", "AI_ObjectLabel", "AI_SceneLabel"]]

    dataSet_id: Optional[str] = None
    data_sources: Optional[List[CI_Citation]] = None
    number_of_labels: Optional[int] = None
    labeling: Optional[List[AI_Labeling]] = None
    training_type: Optional[str] = None
    quality: Optional[List[DataQuality]] = None

    @field_validator("training_type")
    def validate_training_type(cls, v):
        if _validate_training_type(v):
            return v

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return AI_TrainingData(**new_dict)


class AI_TDChangeset(BaseCamelModel):
    type: Literal["AI_TDChangeset"]
    id: str
    change_count: int

    dataset_id: Optional[str] = None
    version: Optional[str] = None
    created_time: Optional[str] = None
    add: Optional[List[Union[AI_TrainingData, "AI_EOTrainingData"]]] = None
    modify: Optional[List[Union[AI_TrainingData, "AI_EOTrainingData"]]] = None
    delete: Optional[List[Union[AI_TrainingData, "AI_EOTrainingData"]]] = None

    @field_validator("created_time")
    def validate_created_time(cls, v):
        return _validate_date(v)

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return AI_TDChangeset(**new_dict)


class TrainingDataset(BaseCamelModel):
    """
    Basic training dataset type
    """

    id: str
    name: str
    description: str
    license: str
    tasks: List[Union[AI_Task, "AI_EOTask"]] = Field(min_length=1)
    data: List[Union[AI_TrainingData, "AI_EOTrainingData"]] = Field(
        min_length=1
    )  # That one should be uri-format
    type: Literal["AI_AbstractTrainingDataset"]

    amount_of_training_data: Optional[int] = None
    classes: Optional[List[NamedValue]] = None
    classification_schema: Optional[str] = None  # That one should be uri-format
    created_time: Optional[str] = None
    data_sources: Optional[List[CI_Citation]] = (
        None  # That string one should be uri-format
    )
    doi: Optional[str] = None
    keywords: Optional[List[str]] = None
    number_of_classes: Optional[int] = None
    providers: Optional[List[str]] = None
    scope: Optional[MD_Scope] = None
    statistics_info: Optional[List[NamedValue]] = None
    updated_time: Optional[str] = None
    version: Optional[str] = None
    labeling: Optional[List[AI_Labeling]] = None
    metrics_in_LIT: Optional[List[AI_MetricsInLiterature]] = None
    quality: Optional[List[DataQuality]] = None
    changesets: Optional[List[AI_TDChangeset]] = None

    @field_validator("created_time")
    def validate_created_time(cls, v):
        if v:
            return _validate_date(v)
        else:
            return v

    @field_validator("updated_time")
    def validate_updated_time(cls, v):
        if v is not None:
            return _validate_date(v)
        else:
            return v

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return TrainingDataset(**new_dict)
