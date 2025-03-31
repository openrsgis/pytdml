import os
from urllib.parse import quote

from pytdml import *

# get training data
from pytdml.io import write_to_json
from pytdml.type import (
    EOTrainingDataset,
    EOTrainingData,
    SceneLabel,
    EOTask,
    EODataSource,
)

td_list = []
image_path = r"D:\TrainingDatasets\WHU-RS19\image"
for root, dirs, files in os.walk(image_path):
    for file in files:
        if file != "Thumbs.db":
            td = EOTrainingData(
                id=file.split(".")[0],
                labels=[SceneLabel(label_class=os.path.relpath(root, image_path))],
                data_url=os.path.join(root, file),
                date_time="2010",
            )
            td_list.append(td)

# generate EO training dataset
whu_rs19 = EOTrainingDataset(
    id="whu_rs19",
    name="WHU_RS19",
    description="WHU-RS19 has 19 classes of remote sensing images scenes obtained from Google Earth",
    tasks=[
        EOTask(
            task_type="Scene Classification",
            description="Structural high-resolution satellite image indexing",
        )
    ],
    data=td_list,
    version="1.0",
    amount_of_training_data=len(td_list),
    created_time="2010",
    updated_time="2010",
    providers=["Wuhan University"],
    keywords=["Remote Sensing", "Scene Classification"],
    data_sources=[
        EODataSource(
            id="google_earth",
            data_type="Optical Image",
            citation="https://earth.google.com/",
            resolution="0.5m",
        )
    ],
    classes=[
        "Airport",
        "Beach",
        "Bridge",
        "Commercial",
        "Desert",
        "Farmland",
        "footballField",
        "Forest",
        "Industrial",
        "Meadow",
        "Mountain",
        "Park",
        "Parking",
        "Pond",
        "Port",
        "railwayStation",
        "Residential",
        "River",
        "Viaduct",
    ],
    number_of_classes=19,
    bands=["red", "green", "blue"],
    image_size="600x600",
)
# write to json
write_to_json(whu_rs19, "whu_rs19.json")
