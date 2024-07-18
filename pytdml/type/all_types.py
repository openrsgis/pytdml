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
import copy
import geojson
from geojson import Feature
from typing import List, Union, Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from pytdml.type._utils import _validate_date, to_camel, _valid_methods, _validate_training_type, _validate_image_format, _validate_evaluation_method_type, to_interior_class, list_to_interior_class


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
        data = self.dict(by_alias=True, exclude_unset=True)
        data_without_none = {k: v for k, v in data.items() if v is not None}
        return data_without_none


class KeyValuePair(BaseCamelModel):
    """
    Key/Value pair type
    """

    key: list
    value: list

    def to_dict(self):
        return self.json()

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
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return NamedValue(**new_dict)


class CI_Date(BaseCamelModel):
    """
    From ISO 19115-1 CI_Date
    """

    date: str
    dateType: str

    @field_validator("date")
    def validate_date(cls, v):
        return _validate_date(v)

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return CI_Date(**new_dict)


class CI_Citation(BaseCamelModel):
    """
    From ISO 19115-1 CI_Citation
    """

    title: str
    alternateTitle: Optional[List[str]] = None
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
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('date'):
            list_to_interior_class(new_dict, 'date', CI_Date)
        if new_dict.__contains__('graphic'):
            list_to_interior_class(new_dict, 'graphic', KeyValuePair)
        if new_dict.__contains__('identifier'):
            list_to_interior_class(new_dict, 'identifier', KeyValuePair)
        return CI_Citation(**new_dict)


class LinearRing(BaseCamelModel):
    """
    gml: LinearRing - NIEM 2.1
    """

    posList: List[float] = Field(min_items=4)

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return LinearRing(**new_dict)


class LinearRing_Object(BaseCamelModel):
    """
    LinearRing Object Type
    """

    linearRing: LinearRing = Field(alias="LinearRing")

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        to_interior_class(new_dict, "linearRing", LinearRing)
        return LinearRing_Object(**new_dict)


class Polygon(BaseCamelModel):
    """
    gml: Polygon - NIEM 2.1
    """

    description: Optional[str] = None
    description_Reference: Optional[str] = None
    identifier: Optional[str] = None
    name: Optional[List[str]] = None
    exterior: Optional[LinearRing_Object] = None
    interior: Optional[List[LinearRing_Object]] = None

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('exterior'):
            to_interior_class(new_dict, 'exterior', LinearRing_Object)
        if new_dict.__contains__('interior'):
            list_to_interior_class(new_dict, 'interior', LinearRing_Object)
        return Polygon(**new_dict)


class MD_Identifier(BaseCamelModel):
    """
    From ISO 19115-1 MD_Identifier
    """

    code: str
    authority: Optional[CI_Citation] = None
    codeSpace: Optional[str] = None
    version: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('authority'):
            list_to_interior_class(new_dict, 'authority', CI_Citation)
        return MD_Identifier(**new_dict)


class MemberName(BaseCamelModel):
    """
    From ISO 19115-1 MD_Identifier
    """

    aName: str
    attributeType: str

    def to_dict(self):
        return self.json()

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
    rangeElement: List[str] = Field(min_items=1)

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return MI_RangeElementDescription(**new_dict)


class MD_Band(BaseCamelModel):
    """
    From ISO 19115-1 MD_Band
    """

    sequenceIdentifier: Optional[MemberName] = None
    description: Optional[str] = None
    name: Optional[List[MD_Identifier]] = None
    maxValue: Optional[float] = None
    minValue: Optional[float] = None
    units: Optional[str] = None
    scaleFactor: Optional[float] = None
    offset: Optional[float] = None
    meanValue: Optional[float] = None
    numberOfValues: Optional[int] = None
    standardDeviation: Optional[float] = None
    otherPropertyType: Optional[float] = None
    otherProperty: Optional[str] = None
    bitsPerValue: Optional[int] = None
    rangeElementDescription: Optional[List[MI_RangeElementDescription]] = None
    boundMax: Optional[float] = None
    boundMin: Optional[float] = None
    boundUnits: Optional[str] = None
    peakResponse: Optional[float] = None
    toneGradation: Optional[int] = None

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
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('sequenceIdentifier'):
            to_interior_class(new_dict, 'sequenceIdentifier', MemberName)
        if new_dict.__contains__('rangeElementDescription'):
            list_to_interior_class(new_dict, 'rangeElementDescription', MI_RangeElementDescription)
        return MD_Band(**new_dict)


class EX_BoundingPolygon(BaseCamelModel):
    """
    From ISO 19115:2003
    """

    polygon: List[Polygon] = Field(min_items=1)
    extentTypeCode: Optional[bool] = None

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('polygon'):
            list_to_interior_class(new_dict, 'polygon', Polygon)
        else:
            print("Some necessary parameters are missing from the provided data.")
            exit(1)
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

    westBoundLongitude: int
    eastBoundLongitude: int
    southBoundLatitude: int
    northBoundLatitude: int

    extentTypeCode: Optional[bool] = None

    def to_dict(self):
        return self.json()

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

    geographicIdentifier: MD_Identifier

    extentTypeCode: Optional[bool] = None

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('geographicIdentifier'):
            to_interior_class(new_dict, 'geographicIdentifier', MD_Identifier)
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

    timePosition: str

    description: Optional[str] = None
    descriptionReference: Optional[str] = None
    identifier: Optional[str] = None
    name: Optional[List[str]] = None
    relatedTime: Optional[List[KeyValuePair]] = None

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('relatedTime'):
            to_interior_class(new_dict, 'relatedTime', KeyValuePair)
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

    beginPosition: str
    endPosition: str

    description: Optional[str] = None
    descriptionReference: Optional[str] = None
    identifier: Optional[str] = None
    name: Optional[List[str]] = None
    duration: Optional[str] = None
    timeInterval: Optional[int] = None
    relatedTime: Optional[List[KeyValuePair]] = None

    @field_validator("beginPosition")
    def validate_begin_position(cls, v):
        return _validate_date(v)

    @field_validator("endPosition")
    def validate_end_position(cls, v):
        return _validate_date(v)

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('relatedTime'):
            to_interior_class(new_dict, 'relatedTime', KeyValuePair)
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
        return self.json()

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
        else:
            print("Some necessary parameters are missing from the provided data.")
            exit(1)
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

    referenceSystemIdentifier: Optional[MD_Identifier] = None
    referenceSystemType: Optional[str] = None

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('referenceSystemIdentifier'):
            to_interior_class(new_dict, 'referenceSystemIdentifier', MD_Identifier)
        return EX_ReferenceSystem(**new_dict)


class VerticalCRS(BaseCamelModel):
    """
    From ISO 19111 edition 2
    """

    identifier: str
    scope: List[str] = Field(min_items=1)
    verticalCS: List[str] = Field(min_items=1)
    verticalDatum: List[str] = Field(min_items=1)

    description: Optional[str] = None
    description_Reference: Optional[str] = None
    name: Optional[List[str]] = None
    remarks: Optional[List[str]] = None
    domain_Of_Validity: Optional[List[str]] = None

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return VerticalCRS(**new_dict)


class EX_VerticalExtent(BaseCamelModel):
    """
    From ISO 19115 SpiralTracker Report
    """

    minimumValue: int
    maximumValue: int

    verticalCRSId: Optional[EX_ReferenceSystem] = None
    verticalCRS: Optional[VerticalCRS] = None

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('verticalCRSId'):
            to_interior_class(new_dict, 'verticalCRSId', EX_ReferenceSystem)
        if new_dict.__contains__('verticalCRS'):
            to_interior_class(new_dict, 'verticalCRS', VerticalCRS)
        return EX_VerticalExtent(**new_dict)


class EX_SpatialTemporalExtent(BaseCamelModel):
    """
    From ISO 19115: 2003 Metadata
    """

    extent: Union[TimeInstant, TimePeriod]
    spatialExtent: Union[EX_BoundingPolygon, EX_GeographicBoundingBox, EX_GeographicDescription]

    verticalExtent: Optional[EX_VerticalExtent] = None

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('extent') and new_dict.__contains__('spatialExtent'):
            extent = new_dict['extent']
            if TimeInstant.can_build_from_data(extent):
                extent = TimeInstant.from_dict(extent)
            else:
                extent = TimePeriod.from_dict(extent)
            new_dict['extent'] = extent

            spatialExtent = new_dict['spatialExtent']
            if EX_BoundingPolygon.can_build_from_data(spatialExtent):
                spatialExtent = EX_BoundingPolygon.from_dict(spatialExtent)
            elif EX_GeographicBoundingBox.can_build_from_data(spatialExtent):
                spatialExtent = EX_GeographicBoundingBox.from_dict(spatialExtent)
            else:
                spatialExtent = EX_GeographicDescription.from_dict(spatialExtent)
            new_dict['spatialExtent'] = spatialExtent
        else:
            print("Some necessary parameters are missing from the provided data.")
            exit(1)

        if new_dict.__contains__('verticalExtent'):
            to_interior_class(new_dict, 'verticalExtent', EX_VerticalExtent)
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
    geographicElement: Optional[List[Union[EX_BoundingPolygon, EX_GeographicBoundingBox, EX_GeographicDescription]]] = None
    temporalElement: Optional[List[Union[EX_TemporalExtent, EX_SpatialTemporalExtent]]] = None
    verticalElement: Optional[List[EX_VerticalExtent]] = None

    def to_dict(self):
        return self.json()

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

        if new_dict.__contains__('verticalElement'):
            list_to_interior_class(new_dict, 'verticalElement',EX_VerticalExtent)
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
#     extent: List[float] = Field(min_items=4)
#
#     def to_dict(self):
#         return self.json()
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
    featureInstances: Optional[str] = None
    attributeInstances: Optional[str] = None
    dataset: Optional[str] = None
    other: Optional[str] = None

    def to_dict(self):
        return self.json()

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
    levelDescription: Optional[MD_ScopeDescription] = Field(None, min_items=4)

    def to_dict(self):
        return self.json()

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

        if new_dict.__contains__('levelDescription'):
            to_interior_class(new_dict, 'levelDescription', MD_ScopeDescription)
        return MD_Scope(**new_dict)


class CI_Telephone(BaseCamelModel):

    number: str
    numberType: Optional[str] = None

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return CI_Telephone(**new_dict)


class CI_Address(BaseCamelModel):

    deliveryPoint: Optional[List[str]] = None
    city: Optional[str] = None
    administrativeArea: Optional[str] = None
    postalCode: Optional[str] = None
    country: Optional[str] = None
    electronicMailAddress: Optional[List[str]] = None

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return CI_Address(**new_dict)


class CI_OnlineResource(BaseCamelModel):

    linkage: str
    protocol: Optional[str] = None
    applicationProfile: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    function: Optional[str] = None
    protocolRequest: Optional[str] = None

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return CI_OnlineResource(**new_dict)


class CI_Contact(BaseCamelModel):

    phone: Optional[List[CI_Telephone]] = None
    address: Optional[List[CI_Address]] = None
    onlineResource: Optional[List[CI_OnlineResource]] = None
    hoursOfService: Optional[List[str]] = None
    contactInstructions: Optional[str] = None
    contactType: Optional[str] = None

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('phone'):
            list_to_interior_class(new_dict, 'phone', CI_Telephone)
        if new_dict.__contains__('address'):
            list_to_interior_class(new_dict, 'address', CI_Address)
        if new_dict.__contains__('onlineResource'):
            list_to_interior_class(new_dict, 'onlineResource', CI_OnlineResource)
        return CI_Contact(**new_dict)


class CI_Individual(BaseCamelModel):

    name: Optional[str] = None
    contactInfo: Optional[List[CI_Contact]] = None
    partyIdentifier: Optional[List[MD_Identifier]] = None
    positionName: Optional[str] = None

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('contactInfo'):
            list_to_interior_class(new_dict, 'contactInfo', CI_Contact)
        if new_dict.__contains__('partyIdentifier'):
            list_to_interior_class(new_dict, 'partyIdentifier', MD_Identifier)
        return CI_Individual(**new_dict)

    def can_build_from_data(data):
        try:
            CI_Individual.from_dict(data)
            return True
        except Exception:
            return False


class CI_Organisation(BaseCamelModel):

    name: Optional[str] = None
    contactInfo: Optional[List[CI_Contact]] = None
    partyIdentifier: Optional[List[MD_Identifier]] = None
    logo: Optional[List[KeyValuePair]] = None
    individual: Optional[List[CI_Individual]] = None

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('contactInfo'):
            list_to_interior_class(new_dict, 'contactInfo', CI_Contact)
        if new_dict.__contains__('partyIdentifier'):
            list_to_interior_class(new_dict, 'partyIdentifier', MD_Identifier)
        if new_dict.__contains__('logo'):
            list_to_interior_class(new_dict, 'logo', KeyValuePair)
        if new_dict.__contains__('individual'):
            list_to_interior_class(new_dict, 'individual', CI_Individual)
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
        return self.json()

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
            print("Some necessary parameters are missing from the provided data.")
            exit(1)

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
    disseminationConstraints: Optional[List[str]] = None

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('addressee'):
            list_to_interior_class(new_dict, 'addressee', CI_Responsibility)
        return MD_Releasability(**new_dict)


class MD_Constraints(BaseCamelModel):

    useLimitation: Optional[List[str]] = None
    constraintApplicationScope: Optional[MD_Scope] = None
    graphic: Optional[List[KeyValuePair]] = None
    reference: Optional[List[CI_Citation]] = None
    releasability: Optional[MD_Releasability] = None
    responsibleParty: Optional[List[CI_Responsibility]] = None

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('constraintApplicationScope'):
            to_interior_class(new_dict, 'constraintApplicationScope', MD_Scope)
        if new_dict.__contains__('graphic'):
            list_to_interior_class(new_dict, 'graphic', KeyValuePair)
        if new_dict.__contains__('reference'):
            list_to_interior_class(new_dict, 'reference', CI_Citation)
        if new_dict.__contains__('releasability'):
            to_interior_class(new_dict, 'releasability', MD_Releasability)
        if new_dict.__contains__('responsibleParty'):
            list_to_interior_class(new_dict, 'responsibleParty', CI_Responsibility)
        return MD_Constraints(**new_dict)


class MD_BrowseGraphic(BaseCamelModel):
    """
    From ISO 19115-1 MD_BrowseGraphic
    """

    file_name: str
    fileDescription: Optional[str] = None
    fileType: Optional[str] = None
    imageConstraints: Optional[List[MD_Constraints]] = None
    linkage: Optional[List[CI_OnlineResource]] = None

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('imageConstraints'):
            list_to_interior_class(new_dict, 'imageConstraints', MD_Constraints)
        if new_dict.__contains__('linkage'):
            list_to_interior_class(new_dict, 'linkage', CI_OnlineResource)
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
#         return self.json()
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
    metrics: List[NamedValue] = Field(min_items=1)

    algorithm: Optional[str] = None

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('metrics'):
            to_interior_class(new_dict, "metrics", NamedValue)
        else:
            print("Some necessary parameters are missing from the provided data.")
            exit(1)
        return AI_MetricsInLiterature(**new_dict)


class AI_Task(BaseCamelModel):
    """
    Basic task type
    """

    id: str
    type: Literal["AI_AbstractTask"]

    datasetId: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return AI_Task(**new_dict)


class AI_EOTask(AI_Task):
    """
    Extended task type for EO training data
    """
    type: Literal["AI_EOTask"]
    taskType: str

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return AI_EOTask(**new_dict)


class AI_Labeler(BaseCamelModel):
    """
    Labeler type
    """

    id: str
    name: str
    type: Literal["AI_Labeler"]

    def to_dict(self):
        return self.json()

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
    methods: List[str] = Field(min_items=1)

    tools: Optional[List[str]] = None

    @field_validator("methods")
    def valid_methods(cls, v):
        return _valid_methods(v)

    def to_dict(self):
        return self.json()

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
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('scope'):
            to_interior_class(new_dict, "scope", MD_Scope)
        else:
            print("Some necessary parameters are missing from the provided data.")
            exit(1)
        if new_dict.__contains__('labelers'):
            list_to_interior_class(new_dict, 'labelers', AI_Labeler)
        if new_dict.__contains__('procedure'):
            to_interior_class(new_dict, "procedure", AI_LabelingProcedure)
        return AI_Labeling(**new_dict)


class MeasureReference(BaseCamelModel):

    measureIdentification: Optional[MD_Identifier] = None
    nameOfMeasure: Optional[List[str]] = None
    measureDescription: Optional[str] = None

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('measureIdentification'):
            list_to_interior_class(new_dict, 'measureIdentification', MD_Identifier)
        return MeasureReference(**new_dict)


class EvaluationMethod(BaseCamelModel):

    name: Optional[str] = None
    evaluationMethodDescription: Optional[str] = None
    evaluationMethodType: Optional[List[str]] = None
    evaluationProcedure: Optional[CI_Citation] = None
    dateTime: Optional[List[str]] = None
    referenceDoc: Optional[List[CI_Citation]] = None
    deductiveSource: Optional[str] = None

    @field_validator("dateTime")
    def validate_date_time(cls, v):
        validated_date = []
        for item in v:
            validated_date.append(_validate_date(item))
        return validated_date

    @field_validator("evaluationMethodType")
    def validate_data_time(cls, v):
        evaluation_method = []
        for item in v:
            evaluation_method.append(_validate_evaluation_method_type(item))
        return evaluation_method

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('referenceDoc'):
            list_to_interior_class(new_dict, 'referenceDoc', CI_Citation)
        if new_dict.__contains__('evaluationProcedure'):
            to_interior_class(new_dict, 'evaluationProcedure', CI_Citation)
        return EvaluationMethod(**new_dict)


class QuantitativeResult(BaseCamelModel):

    value: List[Union[str, object, int, float, list, bool, None]]

    valueUnit: Optional[str] = None
    valueRecordType: Optional[str] = None

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return QuantitativeResult(**new_dict)


class ConformanceResult(BaseCamelModel):

    passBool: bool = Field(alias="pass")
    specification: CI_Citation

    explanation: Optional[List[str]] = None

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('specification'):
            to_interior_class(new_dict, 'specification', CI_Citation)
        else:
            print("Some necessary parameters are missing from the provided data.")
            exit(1)
        return ConformanceResult(**new_dict)


class DescriptiveResult(BaseCamelModel):

    statement: str

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return DescriptiveResult(**new_dict)


class MD_Dimension(BaseCamelModel):

    dimensionName: str
    dimensionSize: int

    resolution: Optional[str] = None
    dimensionTitle: Optional[str] = None
    dimensionDescription: Optional[str] = None

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return MD_Dimension(**new_dict)


class MD_GridSpatialRepresentation(BaseCamelModel):

    numberOfDimensions: int
    cellGeometry: str
    transformationParameterAvailability: bool

    scope: Optional[MD_Scope] = None
    axisDimensionProperties: Optional[List[MD_Dimension]]

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('scope'):
            to_interior_class(new_dict, 'scope', MD_Scope)
        if new_dict.__contains__('axisDimensionProperties'):
            list_to_interior_class(new_dict, 'axisDimensionProperties', MD_Dimension)
        return MD_GridSpatialRepresentation(**new_dict)

    def can_build_from_data(data):
        try:
            MD_GridSpatialRepresentation.from_dict(data)
            return True
        except Exception:
            return False


class MD_GeometricObjects(BaseCamelModel):

    geometricObjectType: str
    geometricObjectCount: Optional[int]

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return MD_GeometricObjects(**new_dict)


class MD_VectorSpatialRepresentation(BaseCamelModel):

    scope: Optional[MD_Scope]
    topologyLevel: Optional[str]
    geometricObjects: Optional[MD_GeometricObjects]

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('scope'):
            to_interior_class(new_dict, "scope", MD_Scope)
        if new_dict.__contains__('geometricObjects'):
            to_interior_class(new_dict, "geometricObjects", MD_GeometricObjects)
        return MD_VectorSpatialRepresentation(**new_dict)

    def can_build_from_data(data):
        try:
            MD_VectorSpatialRepresentation.from_dict(data)
            return True
        except Exception:
            return False


class MD_RangeDimension(BaseCamelModel):

    sequenceIdentifier: Optional[MemberName] = None
    description: Optional[str] = None
    name: Optional[List[MD_Identifier]] = None

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('sequenceIdentifier'):
            to_interior_class(new_dict, "sequenceIdentifier", MemberName)
        if new_dict.__contains__('name'):
            list_to_interior_class(new_dict, "name", MD_Identifier)
        return MD_RangeDimension(**new_dict)


class CoverageResult(BaseCamelModel):

    spatialRepresentationType: str
    resultSpatialRepresentation: Union[MD_GridSpatialRepresentation, MD_VectorSpatialRepresentation]

    resultContent: Optional[List[MD_RangeDimension]] = None
    resultFormat: Optional[str] = None

    def to_dict(self):
        return self.json()

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
            print("Some necessary parameters are missing from the provided data.")
            exit(1)

        if new_dict.__contains__('resultContent'):
            list_to_interior_class(new_dict, "resultContent", MD_RangeDimension)
        return CoverageResult(**new_dict)


class QualityElement(BaseCamelModel):
    """
    From ISO 19157-1 QualityElement
    """

    type: str
    measure: MeasureReference
    evaluationMethod: EvaluationMethod
    result: List[Union[QuantitativeResult, ConformanceResult, DescriptiveResult, CoverageResult]]

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('measure') and new_dict.__contains__('evaluationMethod') and new_dict.__contains__('result'):
            to_interior_class(new_dict, "measure", MeasureReference)
            to_interior_class(new_dict, "evaluationMethod", EvaluationMethod)

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
            print("Some necessary parameters are missing from the provided data.")
            exit(1)
        return QualityElement(**new_dict)


class DataQuality(BaseCamelModel):
    """
    From ISO 19157-1 DataQuality
    """

    type: Literal["DataQuality"]
    scope: MD_Scope
    report: Optional[List[QualityElement]] = None

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('scope'):
            to_interior_class(new_dict, "scope", MD_Scope)
        else:
            print("Some necessary parameters are missing from the provided data.")
            exit(1)
        if new_dict.__contains__('report'):
            list_to_interior_class(new_dict, "report", QualityElement)
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
#     bbox: Optional[List[float]] = Field(min_items=4)


class AI_Label(BaseCamelModel):
    """
    Basic label type
    """

    type: Literal["AI_AbstractLabel"]

    is_negative: Optional[bool] = False  # Optional without default value
    confidence: Optional[float] = Field(1.0, ge=0.0, le=1.0)

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return AI_Label(**new_dict)


class AI_PixelLabel(AI_Label):
    """
    Extended label type for pixel level training data
    """

    type: Literal["AI_PixelLabel"]
    imageURL: List[str] = Field(min_items=1)
    imageFormat: List[str] = Field(min_items=1)

    @field_validator("imageFormat")
    def validate_image_format(cls, v):
        valid_format = []
        for item in v:
            valid_format.append(_validate_image_format(item))
        return valid_format

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return AI_PixelLabel(**new_dict)


class AI_ObjectLabel(AI_Label):
    """
    Extended label type for object level training data
    """

    type: Literal["AI_ObjectLabel"]
    object: Feature
    label_class: str = Field(alias="class")

    dateTime: Optional[str] = None
    bboxType: Optional[str] = None

    @field_validator("dateTime")
    def validate_date_time(cls, v):
        return _validate_date(v)

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('object'):
            new_dict["object"] = geojson.loads(json.dumps(json_dict["object"]))
        else:
            print("Some necessary parameters are missing from the provided data.")
            exit(1)
        return AI_ObjectLabel(**new_dict)


class AI_SceneLabel(AI_Label):
    """
    Extended label type for scene level training data
    """

    type: Literal["AI_SceneLabel"]
    label_class: str = Field(alias="class")

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return AI_SceneLabel(**new_dict)


class AI_TrainingData(BaseCamelModel):
    """
    Basic training data type
    """

    type: Literal["AI_AbstractTrainingData"]
    id: str
    labels: List[Union[AI_Label, AI_PixelLabel, AI_ObjectLabel, AI_SceneLabel]]

    datasetId: Optional[str] = None
    dataSources: Optional[List[CI_Citation]] = None
    numberOfLabels: Optional[int] = None
    labeling: Optional[List[AI_Labeling]] = None
    trainingType: Optional[str] = None
    quality: Optional[List[DataQuality]] = None

    @field_validator("trainingType")
    def validate_training_type(cls, v):
        valid_format = []
        for item in v:
            valid_format.append(_validate_training_type(item))
        return valid_format

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
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
            print("Some necessary parameters are missing from the provided data.")
            exit(1)

        if new_dict.__contains__('dataSources'):
            list_to_interior_class(new_dict, "dataSources", CI_Citation)
        if new_dict.__contains__('labeling'):
            list_to_interior_class(new_dict, "labeling", AI_Labeling)
        if new_dict.__contains__('quality'):
            list_to_interior_class(new_dict, "quality", DataQuality)
        return AI_TrainingData(**new_dict)


class AI_EOTrainingData(AI_TrainingData):
    """
    Extended training data type for EO training data
    """

    type: Literal["AI_EOTrainingData"]
    dataURL: List[str] = Field(min_items=1)  # That one should be uri-format

    extent: Optional[Union[EX_Extent, List[float]]] = None
    dataTime: Optional[List[str]] = None

    @field_validator("dataTime")
    def validate_data_time(cls, v):
        validated_date = []
        for item in v:
            validated_date.append(_validate_date(item))
        return validated_date

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
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
        new_dict['labels'] = labels
        if new_dict.__contains__('dataSources'):
            list_to_interior_class(new_dict, "dataSources", CI_Citation)
        if new_dict.__contains__('labeling'):
            list_to_interior_class(new_dict, "labeling", AI_Labeling)
        if new_dict.__contains__('quality'):
            list_to_interior_class(new_dict, "quality", DataQuality)

        if new_dict.__contains__('extent'):
            extent = new_dict['extent']
            for i in range(len(extent)):
                if EX_Extent.can_build_from_data(extent[i]):
                    extent[i] = EX_Extent.from_dict(extent[i])
                else:
                    pass
            new_dict['extent'] = extent

        return AI_EOTrainingData(**new_dict)


# class Changeset(BaseCamelModel):
#     """
#     Training Data Changeset
#     """
#
#     type: Literal["AI_TDChangeset"]
#     id: str
#     change_count: int
#
#     add: Optional[List[AI_TrainingData]]
#     change_count: Optional[int]
#     dataset_id: Optional[str]
#     delete: Optional[List[AI_TrainingData]]
#     modify: Optional[List[AI_TrainingData]]
#     version: Optional[str]
#     created_time: Optional[str]
#
#     @field_validator("created_time")
#     def validate_created_time(cls, v):
#         return _validate_date(v)
#
#     def to_dict(self):
#         return self.json()
#
#     @staticmethod
#     def from_dict(json_dict):
#         return Changeset(**json_dict)


# class StatisticsInfoType(BaseCamelModel):
#     """
#     Statistics info type
#     """
#
#     key: str
#     value: int
#
#     def to_dict(self):
#         return self.json()
#
#     @staticmethod
#     def from_dict(json_dict):
#         return StatisticsInfoType(**json_dict)
#
#
# class StatisticsInfo(BaseCamelModel):
#     """
#     Statistics info
#     """
#
#     type: Optional[List[StatisticsInfoType]] = Field(min_items=1)
#
#     def to_dict(self):
#         return self.json()
#
#     @staticmethod
#     def from_dict(json_dict):
#         return StatisticsInfo(**json_dict)


class AI_TDChangeset(BaseCamelModel):

    type: Literal["AI_TDChangeset"]
    id: str
    change_count: int

    dataset_id: Optional[str] = None
    version: Optional[str] = None
    created_time: Optional[str] = None
    add: Optional[List[Union[AI_TrainingData, AI_EOTrainingData]]] = None
    modify: Optional[List[Union[AI_TrainingData, AI_EOTrainingData]]] = None
    delete: Optional[List[Union[AI_TrainingData, AI_EOTrainingData]]] = None

    @field_validator("created_time")
    def validate_created_time(cls, v):
        return _validate_date(v)

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
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
    tasks: List[Union[AI_Task, AI_EOTask]] = Field(min_items=1)
    data: List[Union[AI_TrainingData, AI_EOTrainingData]] = Field(min_items=1)  # That one should be uri-format
    type: Literal["AI_AbstractTrainingDataset"]

    amountOfTrainingData: Optional[int] = None
    classes: Optional[List[NamedValue]] = None
    classificationScheme: Optional[str] = None  # That one should be uri-format
    createdTime: Optional[str] = None
    dataSources: Optional[List[CI_Citation]] = None  # That string one should be uri-format
    doi: Optional[str] = None
    keywords: Optional[List[str]] = None
    numberOfClasses: Optional[int] = None
    providers: Optional[List[str]] = None
    scope: Optional[MD_Scope] = None
    statisticsInfo: Optional[List[NamedValue]] = None
    updatedTime: Optional[str] = None
    version: Optional[str] = None
    labeling: Optional[List[AI_Labeling]] = None
    metricsInLIT: Optional[List[AI_MetricsInLiterature]] = None
    quality: Optional[List[DataQuality]] = None
    changesets: Optional[List[AI_TDChangeset]] = None

    @field_validator("createdTime")
    def validate_created_time(cls, v):
        return _validate_date(v)

    @field_validator("updatedTime")
    def validate_updated_time(cls, v):
        if v is not None:
            return _validate_date(v)
        else:
            return v

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
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
            print("Some necessary parameters are missing from the provided data.")
            exit(1)

        if new_dict.__contains__('classes'):
            list_to_interior_class(new_dict, "classes", NamedValue)
        if new_dict.__contains__('dataSources'):
            list_to_interior_class(new_dict, "dataSources", CI_Citation)
        if new_dict.__contains__('scope'):
            to_interior_class(new_dict, "scope", MD_Scope)
        if new_dict.__contains__('statisticsInfo'):
            list_to_interior_class(new_dict, "statisticsInfo", NamedValue)
        if new_dict.__contains__('labeling'):
            list_to_interior_class(new_dict, "labeling", AI_Labeling)
        if new_dict.__contains__('metricsInLIT'):
            list_to_interior_class(new_dict, "metricsInLIT", AI_MetricsInLiterature)
        if new_dict.__contains__('quality'):
            list_to_interior_class(new_dict, "quality", DataQuality)
        if new_dict.__contains__('changesets'):
            list_to_interior_class(new_dict, "changesets", AI_TDChangeset)
        return TrainingDataset(**new_dict)


class EOTrainingDataset(TrainingDataset):
    """
    Extended training dataset type for EO training dataset
    """

    type: Literal["AI_EOTrainingDataset"]
    tasks: List[AI_EOTask] = Field(min_items=1)
    data: List[AI_EOTrainingData] = Field(min_items=1)
    # For Convinience, we allow the user to specify the bands by name

    bands: Optional[List[MD_Band]] = None
    extent: Optional[EX_Extent] = None
    imageSize: Optional[str] = None

    def to_dict(self):
        return self.json()

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        if new_dict.__contains__('tasks') and new_dict.__contains__('data'):
            list_to_interior_class(new_dict, "tasks", AI_EOTask)
            list_to_interior_class(new_dict, "data", AI_EOTrainingData)
        else:
            print("Some necessary parameters are missing from the provided data.")
            exit(1)
        if new_dict.__contains__('bands'):
            list_to_interior_class(new_dict, "bands", MD_Band)
        if new_dict.__contains__('extent'):
            list_to_interior_class(new_dict, "extent", EX_Extent)
        if new_dict.__contains__('classes'):
            list_to_interior_class(new_dict, "classes", NamedValue)
        if new_dict.__contains__('dataSources'):
            list_to_interior_class(new_dict, "dataSources", CI_Citation)
        if new_dict.__contains__('scope'):
            to_interior_class(new_dict, "scope", MD_Scope)
        if new_dict.__contains__('statisticsInfo'):
            list_to_interior_class(new_dict, "statisticsInfo", NamedValue)
        if new_dict.__contains__('labeling'):
            list_to_interior_class(new_dict, "labeling", AI_Labeling)
        if new_dict.__contains__('metricsInLIT'):
            list_to_interior_class(new_dict, "metricsInLIT", AI_MetricsInLiterature)
        if new_dict.__contains__('quality'):
            list_to_interior_class(new_dict, "quality", DataQuality)
        if new_dict.__contains__('changesets'):
            list_to_interior_class(new_dict, "changesets", AI_TDChangeset)
        return EOTrainingDataset(**new_dict)


# if __name__ == '__main__':
    # CI_Citation
    # dict = [
    #     {
    #         "title": "Landsat-8",
    #         "date": [
    #             {
    #                 "date": "2017-01-05",
    #                 "dateType": "q"
    #             },
    #             {
    #                 "date": "2017-02-18",
    #                 "dateType": "1"
    #             },
    #         ]
    #     }
    # ]
    # for i in dict:
    #     print(CI_Citation.from_dict(i))

    # MD_Band
    # dict = [
    #     {
    #         "name": [
    #             {
    #                 "code": "red"
    #             }
    #         ]
    #     },
    #     {
    #         "name": [
    #             {
    #                 "code": "green"
    #             }
    #         ]
    #     },
    #     {
    #         "name": [
    #             {
    #                 "code": "blue"
    #             }
    #         ]
    #     },
    #     {
    #         "name": [
    #             {
    #                 "code": "VH"
    #             }
    #         ]
    #     },
    #     {
    #         "name": [
    #             {
    #                 "code": "VV"
    #             }
    #         ]
    #     },
    #     {
    #         "name": [
    #             {
    #                 "code": "VV/VH"
    #             }
    #         ]
    #     }
    # ]
    # for i in dict:
    #     print(MD_Band.from_dict(i))

    # RangeElementDescription_dic = {
    #     "name": "B3",
    #     "definition": "dwd",
    #     "rangeElement": ["dwa", "wa"]
    # }
    # print(MI_RangeElementDescription.from_dict(RangeElementDescription_dic))

    # scope = {
    #     "level": "12",
    #     "extent": [
    #         [
    #             1560160.0,
    #             5176338.4,
    #             1566661.4,
    #             5179409.2
    #         ]
    #     ]
    #     # "level_description":
    # }
    # scope = {
    #     "level": "12",
    #     "extent": [
    #         {
    #             "description": "feqfqw"
    #         }
    #     ]
    # }
    # print(MD_Scope.from_dict(scope))

    # polygon
    # polygon = {
    #     "description": "test",
    #     "name": [
    #         "dayuan",
    #         "xiaoyuan"
    #     ],
    #     "interior": [
    #         {
    #             "linearRing": {
    #                 "posList": [
    #                     2306.0,
    #                     729.0,
    #                     2330.0,
    #                     729.0
    #                 ]
    #             }
    #         }
    #     ]
    # }
    # print(Polygon.from_dict(polygon))

    # EX_Extent

    # extent = {
    #     "description": "test",
    #     "geographicElement": [
    #         {
    #             "polygon": [{
    #                     "description": "test",
    #                     "name": [
    #                         "dayuan",
    #                         "xiaoyuan"
    #                     ],
    #                     "interior": [
    #                         {
    #                             "linearRing": {
    #                                 "posList": [
    #                                     2306.0,
    #                                     729.0,
    #                                     2330.0,
    #                                     729.0
    #                                 ]
    #                             }
    #                         }
    #                     ]
    #                 }],
    #             "extentTypeCode": True
    #         }
    #     ],
    #     "temporalElement": [
    #         {
    #             "extent": {
    #                 "beginPosition": "cwewc",
    #                 "endPosition": "cwec"
    #             }
    #         }
    #     ],
    #     "verticalElement": [
    #         {
    #             "minimumValue": 12,
    #             "maximumValue": 12
    #         }
    #     ]
    # }
    # print(EX_Extent.from_dict(extent))

    # AI_EOTrainingData
    # data = [
    #     {
    #         "type": "AI_EOTrainingData",
    #         "id": "P0000",
    #         "dataSources": [
    #             {
    #                 "title": "GoogleEarth"
    #             }
    #         ],
    #         "dataURL": [
    #             "train/images/P0000.png"
    #         ],
    #         "numberOfLabels": 444,
    #         "trainingType": "training",
    #         "labels": [
    #             {
    #                 "type": "AI_PixelLabel",
    #                 "imageURL": [
    #                     "train/Instance_masks/P0000_instance_id_RGB.png",
    #                     "train/Semantic_masks/P0000_instance_color_RGB.png"
    #                 ],
    #                 "imageFormat":["image/png", "image/png"]
    #             },
    #             {
    #                 "type": "AI_ObjectLabel",
    #                 "class": "1",
    #                 "bboxType": "Horizontal BBox",
    #                 "object": {
    #                     "type": "Feature",
    #                     "properties": {
    #                         "iscrowd": 0,
    #                         "area": 2580
    #                     },
    #                     "geometry": {
    #                         "type": "Polygon",
    #                         "bbox": [
    #                             244.0,
    #                             1602.0,
    #                             306.0,
    #                             1653.0
    #                         ],
    #                         "coordinates": [
    #                             [
    #                                 [
    #                                     274,
    #                                     1602
    #                                 ],
    #                                 [
    #                                     273,
    #                                     1603
    #                                 ],
    #                                 [
    #                                     272,
    #                                     1603
    #                                 ],
    #                                 [
    #                                     271,
    #                                     1603
    #                                 ],
    #                                 [
    #                                     270,
    #                                     1604
    #                                 ]
    #                             ]
    #                         ]
    #                     }
    #                 }
    #             }
    #         ]
    #     }
    # ]
    # print(AI_EOTrainingData.from_dict(data[0]))

    # AI_TDChangeset

    # changeset = {
    #     "type": "AI_TDChangeset",
    #     "id": "changeset-dota_v1.5",
    #     "datasetId": "dota_v1.5",
    #     "createdTime": "2019-01-01",
    #     "changeCount": 9,
    #     "modify": [
    #         {
    #             "type": "AI_EOTrainingData",
    #             "id": "P1228",
    #             "dataSources": [
    #                 {
    #                     "title": "GF"
    #                 }
    #             ],
    #             "dataURL": [
    #                 "train/images/P1228.png"
    #             ],
    #             "numberOfLabels": 50,
    #             "trainingType": "training",
    #             "labels": [
    #                 {
    #                     "type": "AI_ObjectLabel",
    #                     "class": "ship",
    #                     "object": {
    #                         "type": "Feature",
    #                         "properties": {
    #                             "type": "Object"
    #                         },
    #                         "geometry": {
    #                             "type": "Polygon",
    #                             "coordinates": [
    #                                 [
    #                                     [
    #                                         2306.0,
    #                                         729.0
    #                                     ],
    #                                     [
    #                                         2330.0,
    #                                         729.0
    #                                     ],
    #                                     [
    #                                         2330.0,
    #                                         744.0
    #                                     ],
    #                                     [
    #                                         2306.0,
    #                                         744.0
    #                                     ],
    #                                     [
    #                                         2306.0,
    #                                         729.0
    #                                     ]
    #                                 ]
    #                             ]
    #                         },
    #                         "bboxType": "Horizontal BBox",
    #                         "isDiffDetectable": False
    #                     }
    #                 }
    #             ]
    #         }
    #     ]
    # }
    # print(AI_TDChangeset.from_dict(changeset))

    # EOTrainingDataset

    # dataset = {
    #     "type": "AI_EOTrainingDataset",
    #     "id": "uit_hcd_california_2017",
    #     "name": "UiT HCD California 2017",
    #     "description": "This dataset is composed of two images and a label image.",
    #     "license": "CC BY-SA 4.0",
    #     "version": "1.0",
    #     "amountOfTrainingData": 1,
    #     "createdTime": "2017-01-01",
    #     "providers": [
    #         "LP DAAC",
    #         "ESA"
    #     ],
    #     "classes": [
    #         {
    #             "key": "change",
    #             "value": 1
    #         },
    #         {
    #             "key": "unchanged",
    #             "value": 0
    #         }
    #     ],
    #     "numberOfClasses": 2,
    #     "bands": [
    #         {
    #             "name": [
    #                 {
    #                     "code": "red"
    #                 }
    #             ]
    #         },
    #         {
    #             "name": [
    #                 {
    #                     "code": "green"
    #                 }
    #             ]
    #         },
    #         {
    #             "name": [
    #                 {
    #                     "code": "blue"
    #                 }
    #             ]
    #         },
    #         {
    #             "name": [
    #                 {
    #                     "code": "VH"
    #                 }
    #             ]
    #         },
    #         {
    #             "name": [
    #                 {
    #                     "code": "VV"
    #                 }
    #             ]
    #         },
    #         {
    #             "name": [
    #                 {
    #                     "code": "VV/VH"
    #                 }
    #             ]
    #         }
    #     ],
    #     "imageSize": "2000x3500",
    #     "tasks": [
    #         {
    #             "type": "AI_EOTask",
    #             "id": "uit_hcd_california_2017-task",
    #             "description": "Multi-source images change detection",
    #             "taskType": "http://demo#change_detection"
    #         }
    #     ],
    #     "data": [
    #         {
    #             "type": "AI_EOTrainingData",
    #             "id": "0",
    #             "dataTime": [
    #                 "2017-01-05",
    #                 "2017-02-18"
    #             ],
    #             "dataURL": [
    #                 "t1_L8.png",
    #                 "t2_SAR.png"
    #             ],
    #             "dataSources": [
    #                 {
    #                     "title": "Landsat-8"
    #                 }
    #             ],
    #             "numberOfLabels": 1,
    #             "labels": [
    #                 {
    #                     "type": "AI_PixelLabel",
    #                     "imageURL": [
    #                         "change_label.png"
    #                     ],
    #                     "imageFormat": [
    #                         "image/png"
    #                     ]
    #                 }
    #             ]
    #         }
    #     ]
    # }
    #
    # print(EOTrainingDataset.from_dict(dataset).to_dict())
