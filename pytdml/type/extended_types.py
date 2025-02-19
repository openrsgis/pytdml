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
from pydantic import Field, field_validator

from pytdml.type._utils import _validate_date, _validate_image_format
from pytdml.type.basic_types import AI_Label, TrainingDataset, AI_TrainingData, MD_Band, EX_Extent, AI_Task


class AI_PixelLabel(AI_Label):
    """
    Extended label type for pixel level training data
    """

    type: Literal["AI_PixelLabel"]
    image_url: List[str] = Field(min_length=1, alias='imageURL')
    image_format: List[str] = Field(min_length=1)

    @field_validator("image_format")
    def validate_image_format(cls, v):
        valid_format = []
        for item in v:
            if _validate_image_format(item):
                valid_format.append(item)
        return valid_format

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

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

    date_time: Optional[str] = None
    bbox_type: Optional[str] = None

    @field_validator("object", mode="before")
    def parse_feature(cls, v):
        if isinstance(v, dict):
            # 使用 geojson 库解析字典为 Feature 对象
            return Feature(**v)
        return v

    @field_validator("date_time")
    def validate_date_time(cls, v):
        return _validate_date(v)

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return AI_ObjectLabel(**new_dict)


class AI_SceneLabel(AI_Label):
    """
    Extended label type for scene level training data
    """

    type: Literal["AI_SceneLabel"]
    label_class: str = Field(alias="class")

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return AI_SceneLabel(**new_dict)


class AI_EOTask(AI_Task):
    """
    Extended task type for EO training data
    """
    type: Literal["AI_EOTask"]
    task_type: str

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return AI_EOTask(**new_dict)


class AI_EOTrainingData(AI_TrainingData):
    """
    Extended training data type for EO training data
    """

    type: Literal["AI_EOTrainingData"]
    data_url: List[str] = Field(min_length=1, alias='dataURL')
    labels: List[Union[AI_Label, AI_PixelLabel, AI_ObjectLabel, AI_SceneLabel]]

    extent: Optional[Union[EX_Extent, List[Union[int, float]]]] = None
    data_time: Optional[List[str]] = None

    @field_validator("data_time")
    def validate_data_time(cls, v):
        validated_date = []
        for item in v:
            validated_date.append(_validate_date(item))
        return validated_date

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return AI_EOTrainingData(**new_dict)


class EOTrainingDataset(TrainingDataset):
    """
    Extended training dataset type for EO training dataset
    """

    type: Literal["AI_EOTrainingDataset"]
    tasks: List[AI_EOTask] = Field(min_length=1)
    data: List[AI_EOTrainingData] = Field(min_length=1)
    # For Convinience, we allow the user to specify the bands by name

    bands: Optional[List[MD_Band]] = None
    extent: Optional[Union[EX_Extent, List[Union[int, float]]]] = None
    image_size: Optional[str] = None

    def to_dict(self):
        return self.model_dump(by_alias=True, exclude_none=True)

    @staticmethod
    def from_dict(json_dict):
        new_dict = copy.deepcopy(json_dict)
        return EOTrainingDataset(**new_dict)
