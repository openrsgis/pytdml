import json

from pytdml.io.tdml_readers import read_from_json


def test_read_and_write():
    tdml_path = r"tests/data/WHU-building.json"
    td = read_from_json(tdml_path)
    with open(tdml_path, 'r') as f:
        data = json.load(f)
    assert td.model_dump(by_alias=True,exclude_none=True) == data
