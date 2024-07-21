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

from geojson import Feature
from typing import List, Union, Optional, Literal
from pydantic import Field, field_validator

from pytdml.type._utils import _validate_image_format, _validate_date
from pytdml.type.basic_types import Label, TrainingDataset, TrainingData, Task, MD_Band, Extent, BoundingBox


class AI_PixelLabel(Label):
    """
    Extended label type for pixel level training data
    """
    type: Literal["AI_PixelLabel"]
    image_URL: List[str] = Field(min_length=1)
    image_format: List[str] = Field(min_length=1)

    @field_validator("image_format")
    def validate_image_format(cls, v):
        valid_format = []
        for item in v:
            if _validate_image_format(item):
                valid_format.append(item)
        return valid_format


class ObjectLabel(Label):
    """
    Extended label type for object level training data
    """
    type: Literal["AI_ObjectLabel"]
    object: Feature
    label_class: str = Field(alias="class")

    date_time: Optional[str] = None
    bbox_type: Optional[str] = None

    @field_validator("date_time")
    def validate_date_time(cls, v):
        return _validate_date(v)


class SceneLabel(Label):
    """
    Extended label type for scene level training data
    """
    type: Literal["AI_SceneLabel"]
    label_class: str = Field(alias="class")


class EOTask(Task):
    """
    Extended task type for EO training data
    """
    type: Literal["AI_EOTask"]
    taskType: str


class EOTrainingData(TrainingData):
    """
    Extended training data type for EO training data
    """
    type: Literal["AI_EOTrainingData"]
    dataURL: List[str] = Field(min_length=1)  # That one should be uri-format

    extent: Optional[Union[Extent, List[Union[int, float]]]] = None  # FIXME: formally define the type
    data_time: Optional[List[str]] = None  # FIXME: should be date-time string

    @field_validator("data_time")
    def validate_data_time(cls, v):
        validated_data = []
        for item in v:
            validated_data.append(_validate_date(item))
        return validated_data


class EOTrainingDataset(TrainingDataset):
    """
    Extended training dataset type for EO training dataset
    """
    type: Literal["AI_EOTrainingDataset"]
    # For Convinience, we allow the user to specify the bands by name

    bands: Optional[List[Union[str, MD_Band]]] = None
    extent: Optional[Union[Extent, List[Union[int, float]]]] = None  # FIXME: formally define the type
    imageSize: Optional[str] = None
    tasks: List[EOTask] = Field(min_length=1)
    data: List[EOTrainingData] = Field(min_length=1)
