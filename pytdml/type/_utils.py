# ------------------------------------------------------------------------------
#
# Project: pytdml
# Authors: Boyi Shangguan, Kaixuan Wang, Zhaoyan Wu
# Created: 2023-10-27
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


from datetime import datetime
import re


class InvalidDatetimeError(ValueError):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def _validate_date(v: str) -> str:
    """validate date string and return datetime object

    Args:
        v (str): date string

    Raises:
        ValidationError: date string does not match any allowed format

    Returns:
        str: valid date string
    """
    formats = [
        "%Y-%m-%dT%H:%M:%S",  # date-time
        "%Y-%m-%d",  # date
        "%H:%M:%S",  # time
    ]
    year_pattern = "^(19|20)\\d{2}$"
    year_month_pattern = "^(19|20)\\d{2}-(0[1-9]|1[0-2])$"

    # Try to match date-time, date, or time format
    for fmt in formats:
        try:
            datetime.strptime(v, fmt)
            return v
        except ValueError:
            pass

    # Try to match year or year-month pattern
    if re.match(year_pattern, v) or re.match(year_month_pattern, v):
        return v

    raise InvalidDatetimeError(f"String {v} does not match any allowed format")


def _validate_image_format(image_format: str):
    # Inverse the mimetypes.types_map to get extensions from MIME types
    image_format_list = {
        "image/aces",
        "image/apng",
        "image/avci",
        "image/avcs",
        "image/avif",
        "image/bmp",
        "image/cgm",
        "image/dicom-rle",
        "image/dpx",
        "image/emf",
        "image/example",
        "image/fits",
        "image/g3fax",
        "image/gif",
        "image/heic",
        "image/heic-sequence",
        "image/heif",
        "image/heif-sequence",
        "image/hej2k",
        "image/hsj2",
        "image/ief",
        "image/j2c",
        "image/jls",
        "image/jp2",
        "image/jpeg",
        "image/jph",
        "image/jphc",
        "image/jpm",
        "image/jpx",
        "image/jxr",
        "image/jxrA",
        "image/jxrS",
        "image/jxs",
        "image/jxsc",
        "image/jxsi",
        "image/jxss",
        "image/ktx",
        "image/ktx2",
        "image/naplps",
        "image/png",
        "image/prs.btif",
        "image/prs.pti",
        "image/pwg-raster",
        "image/svg+xml",
        "image/t38",
        "image/tiff",
        "image/tiff-fx",
        "image/tiff; application=geotiff",
        "image/vnd.adobe.photoshop",
        "image/vnd.airzip.accelerator.azv",
        "image/vnd.cns.inf2",
        "image/vnd.dece.graphic",
        "image/vnd.djvu",
        "image/vnd.dwg",
        "image/vnd.dxf",
        "image/vnd.dvb.subtitle",
        "image/vnd.fastbidsheet",
        "image/vnd.fpx",
        "image/vnd.fst",
        "image/vnd.fujixerox.edmics-mmr",
        "image/vnd.fujixerox.edmics-rlc",
        "image/vnd.globalgraphics.pgb",
        "image/vnd.microsoft.icon",
        "image/vnd.mix",
        "image/vnd.ms-modi",
        "image/vnd.mozilla.apng",
        "image/vnd.net-fpx",
        "image/vnd.pco.b16",
        "image/vnd.radiance",
        "image/vnd.sealed.png",
        "image/vnd.sealedmedia.softseal.gif",
        "image/vnd.sealedmedia.softseal.jpg",
        "image/vnd.svf",
        "image/vnd.tencent.tap",
        "image/vnd.valve.source.texture",
        "image/vnd.wap.wbmp",
        "image/vnd.xiff",
        "image/vnd.zbrush.pcx",
        "image/webp",
        "image/wmf",
        "image/x-emf",
        "image/x-wmf",
        "application/json",
        "application/xml",
        "application/xhtml+xml",
        "application/x-netcdf",
        "application/geopackage+sqlite3"
    }
    if image_format in image_format_list:
        return image_format
    else:
        return None


def _valid_methods(labeling_methods: str):
    labeling_methods_list = ["manual", "automatic", "semi-automatic", "unknown"]
    if labeling_methods in labeling_methods_list:
        return labeling_methods
    else:
        return None


def _validate_training_type(training_type: str):
    training_type_list = ["training", "validation", "test", "retraining"]
    if training_type in training_type_list:
        return training_type
    else:
        return None


def _validate_evaluation_method_type(evaluation_method_type: str):
    evaluation_method_type_list = ["directInternal", "directExternal", "indirect"]
    if evaluation_method_type in evaluation_method_type_list:
        return evaluation_method_type
    else:
        return None


def to_camel(string: str) -> str:
    """Change snake_case to camelCase

    Args:
        string (str): snake_case string

    Returns:
        str: camelCase string
    """
    return re.sub(r"_(\w)", lambda match: match.group(1).upper(), string)


def to_interior_class(data_dict, name, class_name):
    new_dic = data_dict[name]
    new_dic = class_name.from_dict(new_dic)
    data_dict[name] = new_dic


def list_to_interior_class(data_dict, name, class_name):
    new_dic = data_dict[name]
    new_dic = [class_name.from_dict(i) for i in new_dic]
    data_dict[name] = new_dic
