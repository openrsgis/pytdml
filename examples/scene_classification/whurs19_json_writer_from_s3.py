import os
from urllib.parse import quote

from pytdml import *

# get training data
from pytdml.io import write_to_json
from pytdml.type import EOTrainingDataset, EOTrainingData, SceneLabel, EOTask

# get training data from s3
s3_client = pytdml.io.S3Client('s3', "your_server", "your_ak", "your_sk")
td_list = []
bucket_name = "my-bucket"
obj_list = s3_client.list_objects(bucket_name=bucket_name, prefix="whu_rs19/")
for obj in obj_list:
    td = EOTrainingData(
        id=obj.split(".")[0],
        labels=[SceneLabel(label_class=obj.split("/")[1])],
        data_url=f"s3://{bucket_name}/{quote(obj)}",
        date_time="2010"
    )
    td_list.append(td)

# generate EO training dataset
whu_rs19 = EOTrainingDataset(
    id='whu_rs19',
    name='WHU_RS19',
    description="WHU-RS19 has 19 classes of remote sensing images scenes obtained from Google Earth",
    tasks=[EOTask(task_type="Scene Classification",
                  description="Structural high-resolution satellite image indexing")],
    data=td_list,
    version="1.0",
    amount_of_training_data=len(td_list),
    created_time="2010",
    updated_time="2010",
    providers=["Wuhan University"],
    keywords=["Remote Sensing", "Scene Classification"],
    data_sources=[EODataSource(
        id="google_earth",
        data_type="Optical Image",
        citation="https://earth.google.com/",
        resolution="0.5m"
    )],
    classes=["Airport", "Beach", "Bridge", "Commercial", "Desert", "Farmland", "footballField", "Forest",
             "Industrial", "Meadow", "Mountain", "Park", "Parking", "Pond", "Port", "railwayStation",
             "Residential", "River", "Viaduct"],
    number_of_classes=19,
    bands=["red", "green", "blue"],
    image_size="600x600"
)
# write to json
write_to_json(whu_rs19, "whu_rs19.json")




