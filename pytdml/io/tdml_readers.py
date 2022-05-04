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
import json

from pytdml.type.basic_types import TrainingDataset
from pytdml.type.extended_types import EOTrainingDataset


def read_from_json(file_path: str):
    """
    Reads a TDML JSON file and returns a TrainingDataset object.
    """
    with open(file_path, "r") as f:
        json_dict = json.load(f)
        # Different kinds of training datasets are supported
        if json_dict["type"] == "TrainingDataset":
            return TrainingDataset.from_dict(json_dict)
        elif json_dict["type"] == "EOTrainingDataset":
            return EOTrainingDataset.from_dict(json_dict)
        else:
            raise ValueError("Unknown TDML type: {}".format(json_dict["type"]))
