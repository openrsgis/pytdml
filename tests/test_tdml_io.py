import pytest
import json
from pydantic import ValidationError, BaseModel, validator

from pytdml.type.extended_types import EOTrainingDataset
from pytdml.io.tdml_writers import write_to_json
from pytdml.io.tdml_readers import read_from_json


def test_read_and_write():
    tdml_path = r"tests/data/WHU-building.json"
    td = read_from_json(tdml_path)
    with open(tdml_path, 'r') as f:
        data = json.load(f)
    assert td.dict(by_alias=True,exclude_none=True) == data
