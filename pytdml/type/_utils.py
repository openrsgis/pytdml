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
import mimetypes





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


def _validate_image_format(mime_type: str):
    # Inverse the mimetypes.types_map to get extensions from MIME types
    extensions_for_mimetype = {v: k for k, v in mimetypes.types_map.items()}
    return mime_type in extensions_for_mimetype


def _valid_methods(labeling_methods: str):
    labeling_methods_list = ["manual", "automatic", "semi-automatic"]
    return labeling_methods in labeling_methods_list

def to_camel(string: str) -> str:
    """Change snake_case to camelCase

    Args:
        string (str): snake_case string

    Returns:
        str: camelCase string
    """
    return re.sub(r"_(\w)", lambda match: match.group(1).upper(), string)
