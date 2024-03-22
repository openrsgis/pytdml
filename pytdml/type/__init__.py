# ------------------------------------------------------------------------------
#
# Project: pytdml
# Authors: Boyi Shangguan, Kaixuan Wang, Zaoyan Wu
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
from .basic_types import BaseCamelModel
from .basic_types import KeyValuePair
from .basic_types import MD_ScopeDescription
from .basic_types import MD_Band
from .basic_types import MD_Scope
from .basic_types import CIDate
from .basic_types import MD_BrowseGraphic
from .basic_types import CICitation
from .basic_types import MD_Identifier
from .basic_types import MetricsPair
from .basic_types import MetricsInLiterature
from .basic_types import Task
from .basic_types import Labeler
from .basic_types import LabelingProcedure
from .basic_types import Labeling
from .basic_types import QualityElement
from .basic_types import DataQuality
from .basic_types import Label
from .basic_types import TrainingData
from .basic_types import Changeset
from .basic_types import StatisticsInfo
from .basic_types import TrainingDataset


from .extended_types import EOTrainingDataset
from .extended_types import EOTrainingData
from .extended_types import SceneLabel
from .extended_types import ObjectLabel
from .extended_types import PixelLabel
from .extended_types import EOTask
