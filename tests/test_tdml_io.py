import json
import requests
import jsonschema

from pytdml.config import BUCKET
from pytdml.io import parse_json
from pytdml.io.S3_reader import S3Client
from pytdml.io.coco_converter import convert_coco_to_tdml
from pytdml.io.yaml_converter import yaml_to_eo_tdml

from pytdml.io.tdml_readers import read_from_json
from pytdml.io.stac_converter import convert_stac_to_tdml

base_url = "https://raw.githubusercontent.com/opengeospatial/TrainingDML-AI_SWG/main/schemas/1.0/json_schema/{}.json"

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
    # remote_schema_url = base_url.format("ai_eoTrainingDataset")
    # response = requests.get(remote_schema_url)
    # remote_schema = response.json()
    # jsonschema.validate(instance=td.to_dict(), schema=remote_schema)
    assert td.to_dict()['type'] == 'AI_EOTrainingDataset'

def test_s3_reader():
    s3 = S3Client(
        resource='s3',
        server= 'https://s3.bcebos.sdhy.omspcloud.com:8443',
        access_key= '1c7ce53fe2ce44ef9071db42ec01fdc1',
        secret_key= '5b1663881aed4423a56cb8b6c784acb1'
    )
    json_data = json.load(s3.get_object(BUCKET.PY, 'examples_1.0/WHU-building.json'))
    td = parse_json(json_data)
    tdml_path = r"tests/data/json/WHU-building.json"
    valid_data = read_from_json(tdml_path).to_dict()
    assert td.to_dict() == valid_data

def test_coco_converter_Object_Detection():
    s3 = S3Client(
        resource='s3',
        server= 'https://s3.bcebos.sdhy.omspcloud.com:8443',
        access_key= '1c7ce53fe2ce44ef9071db42ec01fdc1',
        secret_key= '5b1663881aed4423a56cb8b6c784acb1'
    )
    coco_data = s3.get_object(BUCKET.PY, 'test_data/coco/instances_val2014.json')
    td = convert_coco_to_tdml(coco_data)
    # jsonschema.validate(instance=td.to_dict(), schema=remote_schema)
    assert td.to_dict()['type'] == 'AI_EOTrainingDataset'

def test_coco_converter_Keypoint_Detection():
    s3 = S3Client(
        resource='s3',
        server= 'https://s3.bcebos.sdhy.omspcloud.com:8443',
        access_key= '1c7ce53fe2ce44ef9071db42ec01fdc1',
        secret_key= '5b1663881aed4423a56cb8b6c784acb1'
    )
    coco_data = s3.get_object(BUCKET.PY, 'test_data/coco/person_keypoints_val2014.json')
    td = convert_coco_to_tdml(coco_data)
    # jsonschema.validate(instance=td.to_dict(), schema=remote_schema)
    assert td.to_dict()['type'] == 'AI_EOTrainingDataset'

def test_coco_converter_Panoptic_Segmentation():
    s3 = S3Client(
        resource='s3',
        server= 'https://s3.bcebos.sdhy.omspcloud.com:8443',
        access_key= '1c7ce53fe2ce44ef9071db42ec01fdc1',
        secret_key= '5b1663881aed4423a56cb8b6c784acb1'
    )
    coco_data = s3.get_object(BUCKET.PY, 'test_data/coco/panoptic_val2017.json')
    td = convert_coco_to_tdml(coco_data)
    # jsonschema.validate(instance=td.to_dict(), schema=remote_schema)
    assert td.to_dict()['type'] == 'AI_EOTrainingDataset'

def test_coco_converter_Image_Captioning():
    s3 = S3Client(
        resource='s3',
        server= 'https://s3.bcebos.sdhy.omspcloud.com:8443',
        access_key= '1c7ce53fe2ce44ef9071db42ec01fdc1',
        secret_key= '5b1663881aed4423a56cb8b6c784acb1'
    )
    coco_data = s3.get_object(BUCKET.PY, 'test_data/coco/captions_val2014.json')
    td = convert_coco_to_tdml(coco_data)
    # jsonschema.validate(instance=td.to_dict(), schema=remote_schema)
    assert td.to_dict()['type'] == 'AI_EOTrainingDataset'
