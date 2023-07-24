import json
from io import BytesIO

from datalibrary.s3Client import minio_client as client
from pytdml.type.extended_types import EOTrainingDataset


def read_from_server(name):
    print("reading dataset " + name + "...")
    encode_data = client.get_object("pytdml", "datasetTDEncodes/" + name + ".json")
    json_dict = json.load(BytesIO(encode_data.read()))
    print("parsing dataset " + name + "...")
    if json_dict["type"] == "EOTrainingDataset":
        return EOTrainingDataset.from_dict(json_dict)
    else:
        raise ValueError("Unknown TDML type: {}".format(json_dict["type"]))


def datasets_load():
    data = client.get_object("pytdml", "datasetDescriptions/datasetDescriptions.json")
    dataset_descriptions = json.load(BytesIO(data.read()))
    return dataset_descriptions


def schema_load():
    data = client.get_object("pytdml", "schema/TrainingDML-AI_Schema.json")
    TrainingDML_AI_Schema = json.load(BytesIO(data.read()))
    return TrainingDML_AI_Schema


def task_load():
    return ['Scene Classification', 'Object Detection', 'Semantic Segmentation',
            'Change Detection', '3D Model Reconstruction']