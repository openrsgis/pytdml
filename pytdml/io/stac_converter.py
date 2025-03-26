# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
#
# Project: pytdml
# Authors: Boyi Shangguan, Kaixuan Wang, Zhaoyan Wu
# Created: 2022-05-04
# Modified: 2023-10-27
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

import json
import re
from datetime import datetime
from geojson import Feature
from pystac import Collection
from pytdml.type import EOTrainingDataset, AI_EOTrainingData, AI_ObjectLabel, AI_EOTask


def convert_stac_to_tdml(stac_dataset_path):
    # Reads JSON data in stac format from a given path.
    with open(stac_dataset_path, "r") as file:
        collection_data = json.load(file)
    collection_object = Collection.from_dict(collection_data)
    stac_collection_dataset = collection_object.to_dict(
        include_self_link=False, transform_hrefs=True
    )

    # Reads the necessary attributes from the Collection object and maps them to the EOTrainingDataset object
    collection_version = stac_collection_dataset.get("stac_version")
    collection_id = stac_collection_dataset.get("id")
    collection_description = stac_collection_dataset.get("description")
    collection_license = stac_collection_dataset.get("license")
    collection_bbox = stac_collection_dataset.get("extent").get("spatial").get("bbox")
    collection_interval = (
        stac_collection_dataset.get("extent").get("temporal").get("interval")
    )
    data_time = []
    for item in collection_interval:
        for time in item:
            cleaned_date_time_str = re.sub(r"(\\+00:00|Z)$", "", time)
            date_time_obj = datetime.strptime(
                cleaned_date_time_str, "%Y-%m-%dT%H:%M:%S.%f"
            )
            formatted_date_time_str = date_time_obj.strftime("%Y-%m-%dT%H:%M:%S")
            data_time.append(formatted_date_time_str)

    if len(collection_bbox) == 1:
        collection_extent = collection_bbox[0]
    else:
        collection_extent = [item for bbox in collection_bbox for item in bbox]

    # Reads the necessary attributes from the item object and maps them to the data object
    collection_links = stac_collection_dataset.get("links")
    collection_filtered_links = [
        link for link in collection_links if link.get("rel") == "item"
    ]

    datalist = []
    for link in collection_filtered_links:
        item_path = link.get("href")
        with open(item_path, "r") as item_file:
            stac_item = json.load(item_file)
        link_id = stac_item.get("id")
        link_rel = link.get("rel")
        feature = Feature(**stac_item)
        link_href = [asset["href"] for asset in stac_item.get("assets").values()]

        label = AI_ObjectLabel(
            type="AI_ObjectLabel", object=feature, label_class=link_rel
        )

        data = AI_EOTrainingData(
            type="AI_EOTrainingData",
            id=link_id,
            labels=[label],
            data_url=link_href,
            data_time=data_time,
        )
        datalist.append(data)

    # Reads the unnecessary attributes from the Collection object and maps them to the EOTrainingDataset object
    collection_name = stac_collection_dataset.get("title")

    tasks = [
        AI_EOTask(
            task_type="STAC",
            id=str(collection_id) + "_task",
            dataset_id=str(collection_id),
            type="AI_EOTask",
        )
    ]

    dataset = EOTrainingDataset(
        # necessary attributes
        id=str(collection_id),
        name=collection_name,
        description=collection_description,
        license=collection_license,
        tasks=tasks,
        data=datalist,
        type="AI_EOTrainingDataset",
        # unnecessary attributes
        version=collection_version,
        extent=collection_extent,
    )

    return dataset
