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
from .basic_types import NamedValue
from .basic_types import CI_Date
from .basic_types import CI_Citation
from .basic_types import LinearRing
from .basic_types import LinearRing_Object
from .basic_types import Polygon
from .basic_types import MD_Identifier
from .basic_types import MemberName
from .basic_types import MI_RangeElementDescription
from .basic_types import MD_Band
from .basic_types import EX_BoundingPolygon
from .basic_types import EX_GeographicBoundingBox
from .basic_types import EX_GeographicDescription
from .basic_types import TimeInstant
from .basic_types import TimePeriod
from .basic_types import EX_TemporalExtent
from .basic_types import EX_ReferenceSystem
from .basic_types import VerticalCRS
from .basic_types import EX_VerticalExtent
from .basic_types import EX_SpatialTemporalExtent
from .basic_types import EX_Extent
from .basic_types import MD_ScopeDescription
from .basic_types import MD_Scope
from .basic_types import CI_Telephone
from .basic_types import CI_Address
from .basic_types import CI_OnlineResource
from .basic_types import CI_Contact
from .basic_types import CI_Individual
from .basic_types import CI_Organisation
from .basic_types import CI_Responsibility
from .basic_types import MD_Releasability
from .basic_types import MD_Constraints
from .basic_types import MD_BrowseGraphic
from .basic_types import AI_MetricsInLiterature
from .basic_types import AI_Task
from .basic_types import AI_Labeler
from .basic_types import AI_LabelingProcedure
from .basic_types import AI_Labeling
from .basic_types import MeasureReference
from .basic_types import EvaluationMethod
from .basic_types import QuantitativeResult
from .basic_types import ConformanceResult
from .basic_types import DescriptiveResult
from .basic_types import MD_Dimension
from .basic_types import MD_GridSpatialRepresentation
from .basic_types import MD_GeometricObjects
from .basic_types import MD_VectorSpatialRepresentation
from .basic_types import MD_RangeDimension
from .basic_types import CoverageResult
from .basic_types import QualityElement
from .basic_types import DataQuality
from .basic_types import AI_Label
from .basic_types import AI_TrainingData
from .basic_types import AI_TDChangeset
from .basic_types import TrainingDataset
from .extended_types import AI_PixelLabel
from .extended_types import AI_ObjectLabel
from .extended_types import AI_SceneLabel
from .extended_types import AI_EOTask
from .extended_types import AI_EOTrainingData
from .extended_types import EOTrainingDataset
