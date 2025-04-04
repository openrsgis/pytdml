# ------------------------------------------------------------------------------
#
# Project: pytdml
# Authors: Shuaiqi Liu
# Created: 2023-02-04
# Email: sqi_liu@whu.edu.cn
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
import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

from pytdml.ml.ml_operators import collate_fn
from torchvision.transforms import transforms
import torch.utils.data as data
from datalibrary.pipeline import PipeLine
from datalibrary.datasetcollection import EOTrainingDatasetCollection, Task
import pytdml.ml.object_transforms as transform_target

transform = transforms.Compose(  # transform for the dataset
    [
        transforms.ToTensor(),
        transforms.CenterCrop(224),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),  # random flip
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalize
    ]
)

target_transform = transform_target.Compose(
    [
        transform_target.ToTensor(),
        transform_target.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transform_target.RandomResize((512, 512)),
    ]
)

path = "."


def datasets_for_scene_task():
    ds_lib = EOTrainingDatasetCollection()
    ds_lib.dataset_list(Task.scene_classification, ["Airport"])
    whurs19_ml = ds_lib["WHU-RS19"]

    print("Load training dataset: " + str(whurs19_ml.name))
    print("Number of training samples: " + str(whurs19_ml.amount_of_training_data))
    print("Number of classes: " + str(whurs19_ml.number_of_classes))

    # across dataset
    rsd46_ml = ds_lib["RSD46-WHU"]
    my_dataset_td = ds_lib.training_data_collection(
        Task.scene_classification, [rsd46_ml, whurs19_ml], ["Airport"]
    )

    my_pipeline = PipeLine(my_dataset_td, path)
    # my_dataset = my_pipeline.torch_dataset(download=False, transform=transform)
    my_data_pipe = my_pipeline.torch_data_pipe(transform=transform)

    dataloader = data.DataLoader(my_data_pipe, batch_size=4, num_workers=4)

    for i, batch in enumerate(dataloader):
        print(batch)
        if i == 3:
            break


def datasets_for_object_task():
    ds_lib = EOTrainingDatasetCollection()
    ds_lib.dataset_list(Task.object_detection)
    dota_ml = ds_lib["AIR-SARShip"]

    print("Load training dataset: " + str(dota_ml.tasks))
    print("Number of training samples: " + str(dota_ml.amount_of_training_data))
    print("Number of classes: " + str(dota_ml.number_of_classes))

    # across dataset
    # rsod_ml = ds_lib["RSOD"]
    # my_dataset_TD = ds_lib.training_data_collection(Task.object_detection, [dota_ml, rsod_ml], ["Plane"])
    # print(len(my_dataset_TD.data))
    my_pipeline = PipeLine(dota_ml, path, crop=(500, 0, 0))
    # NWPU_VHR10_torchPipe = my_pipeline.torch_dataset(download=True, transform=target_transform)
    my_torchPipe = my_pipeline.torch_data_pipe(transform=target_transform)
    dataloader = data.DataLoader(
        my_torchPipe, batch_size=4, num_workers=1, collate_fn=collate_fn
    )

    for i, batch in enumerate(dataloader):
        print(batch)
        if i == 3:
            break
    # print("start..")
    # ds_lib = EOTrainingDatasetCollection()
    # ds_lib.dataset_list(Task.object_detection)
    # dota_ml = ds_lib["DOTA-v2.0"]

    # print("Load training dataset: " + str(dota_ml.tasks))
    # print("Number of training samples: " + str(dota_ml.amount_of_training_data))
    # print("Number of classes: " + str(dota_ml.number_of_classes))
    #
    # # across dataset
    # dior_ml  =ds_lib["DIOR"]
    #
    # rsod_ml = ds_lib["RSOD"]
    # my_dataset_TD = ds_lib.training_data_collection(Task.object_detection, [dota_ml, rsod_ml, dior_ml], ["Plane"])
    # print(len(my_dataset_TD.data))
    # my_pipeline = PipeLine(my_dataset_TD, path, (512, 0.25, 0.25))
    # NWPU_VHR10_torchPipe = my_pipeline.torch_dataset(download=True, transform=target_transform)
    # # my_torchPipe = my_pipeline.torch_data_pipe(transform=target_transform)
    # # dataloader = DataLoader2(NWPU_VHR10_torchPipe, batch_size=4, num_workers=4, collate_fn=collate_fn)
    #
    # # for batch in dataloader:
    #  #   print(batch)


def datasets_for_segmentation_task():
    ds_lib = EOTrainingDatasetCollection()
    ds_lib.dataset_list(Task.semantic_segmentation, ["Building Area"])
    AISD_ml = ds_lib["waterExtraction"]

    print("Load training dataset: " + AISD_ml.name)
    print("Number of training samples: " + str(AISD_ml.amount_of_training_data))
    print("Number of classes: " + str(AISD_ml.number_of_classes))

    # across dataset
    # building_extraction_ml = ds_lib["buildingExtraction"]
    # my_dataset_TD = ds_lib.training_data_collection(Task.semantic_segmentation, [AISD_ml, building_extraction_ml], ["Building Area"])
    building_extraction_ml = PipeLine(AISD_ml, path)
    # NWPU_VHR10_torchPipe = building_extraction_ml.torch_dataset(download=True, transform=transform)
    building_extraction_torchPipe = building_extraction_ml.torch_data_pipe(
        transform=transform
    )
    dataloader = data.DataLoader(
        building_extraction_torchPipe, batch_size=4, num_workers=4
    )
    for i, batch in enumerate(dataloader):
        print(batch)
        if i == 3:
            break


def datasets_for_change_task():
    ds_lib = EOTrainingDatasetCollection()
    ds_lib.dataset_list(Task.change_detection)
    sysu_ml = ds_lib["SOUTHGIS Remote Sensing Building Change Detection DataSet"]

    print("Load training dataset: " + sysu_ml.name)
    print("Number of training samples: " + str(sysu_ml.amount_of_training_data))
    print("Number of classes: " + str(sysu_ml.number_of_classes))
    #
    # # across dataset
    # be_ml = ds_lib["buildingExtraction"]
    # my_dataset_TD = ds_lib.training_data_collection(Task.semantic_segmentation, [AISD_ml, be_ml], ["Building Area"])
    # print(len(my_dataset_TD.data))
    my_pipeline = PipeLine(sysu_ml, path)
    # my_torchPipe = my_pipeline.torch_dataset(download=True, transform=transform)
    my_torchPipe = my_pipeline.torch_data_pipe(transform=transform)
    dataloader = data.DataLoader(my_torchPipe, batch_size=4, num_workers=4)
    for i, batch in enumerate(dataloader):
        print(batch)
        if i == 3:
            break


if __name__ == "__main__":
    datasets_for_change_task()
