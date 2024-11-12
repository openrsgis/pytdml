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
from pytdml.type._utils import _validate_date, to_camel, _valid_methods, _validate_training_type, _validate_evaluation_method_type, to_interior_class, list_to_interior_class


class BaseCamelModel(BaseModel):
    """
    Basic model with camel case alias
    Since python use snake case as default
    We need to convert it to camel case for JSON
    """
    model_config: TypedDict = {
        'alias_generator': to_camel,
        'populate_by_name': True,
        'arbitrary_types_allowed': True
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
        key = json_dict.keys()
        value = json_dict.values()
        key_value_pair = {"key": key, "value": value}
        return KeyValuePair(**key_value_pair)


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

    def can_build_from_data(data):
        try:
            EX_BoundingPolygon.from_dict(data)
            return True
        except Exception:
            return False


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

    def can_build_from_data(data):
        try:
            EX_GeographicBoundingBox.from_dict(data)
            return True
        except Exception:
            return False


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

    def can_build_from_data(data):
        try:
            EX_GeographicDescription.from_dict(data)
            return True
        except Exception:
            return False


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

    def can_build_from_data(data):
        try:
            TimeInstant.from_dict(data)
            return True
        except Exception:
            return False


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

    def can_build_from_data(data):
        try:
            TimePeriod.from_dict(data)
            return True
        except Exception:
            return False


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
        if new_dict.__contains__('extent'):
            extent = new_dict['extent']
            if TimeInstant.can_build_from_data(extent):
                extent = TimeInstant.from_dict(extent)
            else:
                extent = TimePeriod.from_dict(extent)
            new_dict['extent'] = extent
        return EX_TemporalExtent(**new_dict)

    def can_build_from_data(data):
        try:
            EX_TemporalExtent.from_dict(data)
            return True
        except Exception:
            return False


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
    spatial_extent: Union[EX_BoundingPolygon, EX_GeographicBoundingBox, EX_GeographicDescription]

    vertical_extent: Optional[EX_VerticalExtent] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('extent'):
            extent = new_dict['extent']
            if TimeInstant.can_build_from_data(extent):
                extent = TimeInstant.from_dict(extent)
            else:
                extent = TimePeriod.from_dict(extent)
            new_dict['extent'] = extent
        if new_dict.__contains__('spatialExtent'):
            spatial_extent = new_dict['spatialExtent']
            if EX_BoundingPolygon.can_build_from_data(spatial_extent):
                spatial_extent = EX_BoundingPolygon.from_dict(spatial_extent)
            elif EX_GeographicBoundingBox.can_build_from_data(spatial_extent):
                spatial_extent = EX_GeographicBoundingBox.from_dict(spatial_extent)
            else:
                spatial_extent = EX_GeographicDescription.from_dict(spatial_extent)
            new_dict['spatialExtent'] = spatial_extent
        return EX_SpatialTemporalExtent(**new_dict)

    def can_build_from_data(data):
        try:
            EX_SpatialTemporalExtent.from_dict(data)
            return True
        except Exception:
            return False


class EX_Extent(BaseCamelModel):
    """
    From ISO 19115: 2003 Metadata
    """

    description: Optional[str] = None
    geographic_element: Optional[List[Union[EX_BoundingPolygon, EX_GeographicBoundingBox, EX_GeographicDescription]]] = None
    temporal_element: Optional[List[Union[EX_TemporalExtent, EX_SpatialTemporalExtent]]] = None
    vertical_element: Optional[List[EX_VerticalExtent]] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('geographicElement'):
            geographic_element = new_dict['geographicElement']
            for i in range(len(geographic_element)):
                if EX_BoundingPolygon.can_build_from_data(geographic_element[i]):
                    geographic_element[i] = EX_BoundingPolygon.from_dict(geographic_element[i])
                elif EX_GeographicBoundingBox.can_build_from_data(geographic_element[i]):
                    geographic_element[i] = EX_GeographicBoundingBox.from_dict(geographic_element[i])
                else:
                    geographic_element[i] = EX_GeographicDescription.from_dict(geographic_element[i])
            new_dict['geographicElement'] = geographic_element

        if new_dict.__contains__('temporalElement'):
            temporal_element = new_dict['temporalElement']
            for i in range(len(temporal_element)):
                if EX_TemporalExtent.can_build_from_data(temporal_element[i]):
                    temporal_element[i] = EX_TemporalExtent.from_dict(temporal_element[i])
                else:
                    temporal_element[i] = EX_SpatialTemporalExtent.from_dict(temporal_element[i])
            new_dict['temporalElement'] = temporal_element
        return EX_Extent(**new_dict)

    def can_build_from_data(data):
        try:
            EX_Extent.from_dict(data)
            return True
        except Exception:
            return False


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
    level_description: Optional[MD_ScopeDescription] = Field(None, min_length=4)

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('extent'):
            extent = new_dict['extent']
            for i in range(len(extent)):
                if EX_Extent.can_build_from_data(extent[i]):
                    extent[i] = EX_Extent.from_dict(extent[i])
                else:
                    pass
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

    def can_build_from_data(data):
        try:
            CI_Individual.from_dict(data)
            return True
        except Exception:
            return False


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

    def can_build_from_data(data):
        try:
            CI_Individual.from_dict(data)
            return True
        except Exception:
            return False


class CI_Responsibility(BaseCamelModel):

    role: str
    party: List[Union[CI_Individual, CI_Organisation]]

    extent: Optional[List[Union[EX_Extent, List[float]]]] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('party'):
            party = new_dict['party']
            for i in range(len(party)):
                if CI_Individual.can_build_from_data(party[i]):
                    party[i] = CI_Individual.from_dict(party[i])
                else:
                    party[i] = CI_Organisation.from_dict(party[i])
                new_dict['party'] = party
        else:
            print("Some necessary parameters of party are not provided.")
            exit()

        if new_dict.__contains__('extent'):
            extent = new_dict['extent']
            for i in range(len(extent)):
                if EX_Extent.can_build_from_data(extent[i]):
                    extent[i] = EX_Extent.from_dict(extent[i])
                else:
                    pass
                new_dict['extent'] = extent
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

    def can_build_from_data(data):
        try:
            MD_GridSpatialRepresentation.from_dict(data)
            return True
        except Exception:
            return False


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

    def can_build_from_data(data):
        try:
            MD_VectorSpatialRepresentation.from_dict(data)
            return True
        except Exception:
            return False


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
    result_spatial_representation: Union[MD_GridSpatialRepresentation, MD_VectorSpatialRepresentation]

    result_content: Optional[List[MD_RangeDimension]] = None
    result_format: Optional[str] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('resultSpatialRepresentation'):
            representation = new_dict['resultSpatialRepresentation']
            if MD_GridSpatialRepresentation.can_build_from_data(representation):
                representation = MD_GridSpatialRepresentation.from_dict(representation)
            else:
                representation = MD_VectorSpatialRepresentation.from_dict(representation)
            new_dict['resultSpatialRepresentation'] = representation
        else:
            print("Some necessary parameters of resultSpatialRepresentation are not provided.")
            exit(0)
        return CoverageResult(**new_dict)


class QualityElement(BaseCamelModel):
    """
    From ISO 19157-1 QualityElement
    """

    type: str
    measure: MeasureReference
    evaluation_method: EvaluationMethod
    result: List[Union[QuantitativeResult, ConformanceResult, DescriptiveResult, CoverageResult]]

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('result'):
            result = new_dict['result']
            for i in range(len(result)):
                if QuantitativeResult.can_build_from_data(result[i]):
                    result[i] = QuantitativeResult.from_dict(result[i])
                elif ConformanceResult.can_build_from_data(result[i]):
                    result[i] = ConformanceResult.from_dict(result[i])
                elif DescriptiveResult.can_build_from_data(result[i]):
                    result[i] = DescriptiveResult.from_dict(result[i])
                else:
                    result[i] = CoverageResult.from_dict(result[i])
            new_dict['resultSpatialRepresentation'] = result
        else:
            print("Parameter \"result\" must be provided.")
            exit(0)
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

    dataset_id: Optional[str] = None
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
        from pytdml.type.extended_types import AI_PixelLabel, AI_ObjectLabel, AI_SceneLabel
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('labels'):
            labels = new_dict['labels']
            for i in range(len(labels)):
                if labels[i]["type"] == "AI_AbstractLabel":
                    labels[i] = AI_Label.from_dict(labels[i])
                elif labels[i]["type"] == "AI_PixelLabel":
                    labels[i] = AI_PixelLabel.from_dict(labels[i])
                elif labels[i]["type"] == "AI_ObjectLabel":
                    labels[i] = AI_ObjectLabel.from_dict(labels[i])
                else:
                    labels[i] = AI_SceneLabel.from_dict(labels[i])
            new_dict['resultSpatialRepresentation'] = labels
        else:
            print("Parameter \"labels\" must be provided.")
            exit(0)
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
        from pytdml.type.extended_types import AI_EOTrainingData
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('add'):
            add = new_dict['add']
            for i in range(len(add)):
                if add[i]["type"] == "AI_EOTrainingData":
                    add[i] = AI_EOTrainingData.from_dict(add[i])
                else:
                    add[i] = AI_TrainingData.from_dict(add[i])
            new_dict['add'] = add

        if new_dict.__contains__('modify'):
            modify = new_dict['modify']
            for i in range(len(modify)):
                if modify[i]["type"] == "AI_EOTrainingData":
                    modify[i] = AI_EOTrainingData.from_dict(modify[i])
                else:
                    modify[i] = AI_TrainingData.from_dict(modify[i])
            new_dict['modify'] = modify

        if new_dict.__contains__('delete'):
            delete = new_dict['delete']
            for i in range(len(delete)):
                if AI_EOTrainingData.can_build_from_data(delete[i]):
                    delete[i] = AI_EOTrainingData.from_dict(delete[i])
                else:
                    delete[i] = AI_TrainingData.from_dict(delete[i])
            new_dict['delete'] = delete
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
    data: List[Union[AI_TrainingData, "AI_EOTrainingData"]] = Field(min_length=1)  # That one should be uri-format
    type: Literal["AI_AbstractTrainingDataset"]

    amount_of_trainingData: Optional[int] = None
    classes: Optional[List[NamedValue]] = None
    classification_scheme: Optional[str] = None  # That one should be uri-format
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
    labeling: Optional[List[AI_Labeling]] = None
    metrics_in_LIT: Optional[List[AI_MetricsInLiterature]] = None
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

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        from pytdml.type.extended_types import AI_EOTask, AI_EOTrainingData
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('tasks') and new_dict.__contains__('data'):
            tasks = new_dict['tasks']
            for i in range(len(tasks)):
                if tasks[i]["type"] == "AI_EOTask":
                    tasks[i] = AI_EOTask.from_dict(tasks[i])
                else:
                    tasks[i] = AI_Task.from_dict(tasks[i])
            new_dict['tasks'] = tasks

            data = new_dict['data']
            for i in range(len(data)):
                if data[i]["type"] == "AI_EOTrainingData":
                    data[i] = AI_EOTrainingData.from_dict(data[i])
                else:
                    data[i] = AI_TrainingData.from_dict(data[i])
            new_dict['data'] = data
        else:
            print("Parameter \"tasks\" and \"data\" must be provided.")
            exit()
        return TrainingDataset(**new_dict)
