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
from typing import Union
from pytdml.type.basic_types import TrainingDataset
from pytdml.utils import remove_empty


# def write_to_json(td: TrainingDataset, file_path: str, indent: Union[None, int, str] = 4):
#     """
#     Writes a TrainingDataset to a JSON file.
#     """
#     with open(file_path, "w", encoding='utf-8') as f:
#         json.dump(remove_empty(td.to_dict()), f, indent=indent, ensure_ascii=False)

def write_to_json(td: TrainingDataset, file_path: str, indent: Union[None, int, str] = 4):
    """
    Writes a TrainingDataset to a JSON file.
    """
    with open(file_path, "w", encoding='utf-8') as f:
        json.dump(td.dict(by_alias=True,exclude_none=True), f, indent=indent, ensure_ascii=False)
        # json.dump(remove_empty(td.dict()), f, indent=indent, ensure_ascii=False)
