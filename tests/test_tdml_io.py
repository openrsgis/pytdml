import pytest
import json
from pydantic import ValidationError, BaseModel, validator
from pytdml.yaml_to_tdml import yaml_to_eo_tdml

from pytdml.type.extended_types import EOTrainingDataset
from pytdml.io.tdml_writers import write_to_json
from pytdml.io.tdml_readers import read_from_json
from pytdml.io.stac_converter import convert_stac_to_tdml


def test_read_and_write():
    tdml_path = r"tests/data/json/WHU-building.json"
    td = read_from_json(tdml_path)
    with open(tdml_path, 'r') as f:
        data = json.load(f)
    assert td.to_dict() == data

def test_yaml_to_eo_tdml():
    yaml_path = r"tests/data/yaml/UiT_HCD_California_2017.yml"
    tdml_path = r"tests/data/json/UiT_HCD_California_2017.json"
    td = yaml_to_eo_tdml(yaml_path)
    with open(tdml_path, 'r') as f:
        data = json.load(f)
    assert td.to_dict() == data

def test_convert_stac_to_tdml():
    stac_file_path = r"tests/data/stac/collection.json"
    td = convert_stac_to_tdml(stac_file_path)
    assert td.to_dict().get("type") == "AI_EOTrainingDataset"
