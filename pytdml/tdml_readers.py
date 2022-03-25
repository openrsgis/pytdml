# ------------------------------------------------------------------------------
#
# Project: pytdml
#
# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
import json

from .basic_types import TrainingDataset
from .extended_types import EOTrainingDataset


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
