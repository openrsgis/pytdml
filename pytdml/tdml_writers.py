# ------------------------------------------------------------------------------
#
# Project: pytdml
#
# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------

import json
from typing import Union
from .basic_types import TrainingDataset
from .utils import remove_empty


def write_to_json(td: TrainingDataset, file_path: str, indent: Union[None, int, str] = 4):
    """
    Writes a TrainingDataset to a JSON file.
    """
    with open(file_path, "w") as f:
        json.dump(remove_empty(td.to_dict()), f, indent=indent)
