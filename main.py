# import os
# import torch
# from torchvision import transforms
# from torch.utils.data import DataLoader2, DataLoader
#
# import pytdml
# from datalibrary.datasetcollection import EOTrainingDatasetCollection
# from pytdml.io import read_from_json
# from pytdml.ml.tdml_torch import BaseTransform
# from pytdml.type import EOTrainingDataset, EOTask
# import pytdml.ml.object_transforms as transform_target
# from pytdml.ml.ml_operators import collate_fn
from datalibrary.s3Client import minio_client

# transform = transforms.Compose(  # transform for the dataset
#     [
#         transforms.ToTensor(),
#         transforms.CenterCrop(224),
#         transforms.RandomCrop(224),
#         transforms.RandomHorizontalFlip(),  # random flip
#         # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalize
#
#     ]
# )
#
# target_transform = transform_target.Compose([
#     transform_target.ToTensor(),
#     transform_target.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     transform_target.RandomResize((512, 512))
# ])
#
# path = r"D:\LiuSQi\Project\pytdml-main\pytdml-main"
#
#
# def datasetsForSceneTask():
#     dataset_library = EOTrainingDatasetCollection()
#     # 获取场景分类数据集资源目录
#     training_datasets_list = dataset_library.dataset_list('Scene Classification')
#
#     # 获取目录资源中RS-C11数据集的 TDML 编码
#     RS_C11_tdml_encoding = EOTrainingDatasetCollection()['rs sensetime']
#
#     # 或者使用发布在网络上或本地的 TDML encoding 文件
#     # AISD = dataset_library.tdml_from_url("http://125.220.153.26/tdml/AISD.json")
#     # print("classes of AISD: " + str(AISD.classes))
#
#     # 获取RS-C11数据集的元数据信息
#     print("Load training dataset: " + RS_C11_tdml_encoding.name)
#     print("Number of training samples: " + str(RS_C11_tdml_encoding.amount_of_training_data))
#     print("Number of classes: " + str(RS_C11_tdml_encoding.number_of_classes))
#
#     # 加载为可供调用的数据集类
#     RS_C11 = dataset_library.sceneDataset(RS_C11_tdml_encoding, root=".", download=False,
#                                           transform=transform)
#
#     print("classes_to_idx: " + str(RS_C11.class_to_idx()))
#     # 跨数据集的数据使用
#     # 选定类别
#     # selected_classes = ["Harbor", "Grass"]
#     # training_datasets_list = dataset_library.dataset_list(task_type='Scene Classification',
#     #                                                       classes=selected_classes)
#     # # 加载目录中两种数据集的 TDML 编码
#     # RS_C11_tdml_encoding = EOTrainingDatasetCollection()['RS-C11']
#     # AID_tdml_encoding = EOTrainingDatasetCollection()['AID']
#     # # 将数据封装为数据集类
#     # my_datasets = EOTrainingDatasetCollection().sceneDataset([AID_tdml_encoding, RS_C11_tdml_encoding], ".",
#     #                                                          selected_classes, False, transform=transform)
#     # 加载数据
#     dataloader = DataLoader2(RS_C11, batch_size=4, num_workers=4)
#
#     for batch in dataloader:
#         print(batch)
#
#
# def datasetsForScenePipe():
#     rs_sensetime_tdml_encoding = EOTrainingDatasetCollection()['rs sensetime']
#     AIDDataPipe = EOTrainingDatasetCollection().sceneDataPipe(rs_sensetime_tdml_encoding, ".", transform=transform)
#     dataloader = DataLoader2(AIDDataPipe, batch_size=4, num_workers=4)
#
#     for batch in dataloader:
#         print(batch)
#
#
# def datasetsForObjectTask():
#     NWPU_VHR10_TD = EOTrainingDatasetCollection()["DOTA-v2.0"]
#
#     NWPU_VHR10 = EOTrainingDatasetCollection().objectDataPipe(NWPU_VHR10_TD, path, crop=(800, 0.25, 0.25),
#                                                              transform=target_transform)
#     dataloader = DataLoader2(NWPU_VHR10, batch_size=4, num_workers=4, collate_fn=collate_fn)
#
#     for batch in dataloader:
#         pass
#
#
# def datasetsForSegmentationTask():
#     AISD_TD = EOTrainingDatasetCollection()["AISD"]
#     print("Load training dataset: " + AISD_TD.name)
#     print("Number of training samples: " + str(AISD_TD.amount_of_training_data))
#     print("Number of classes: " + str(AISD_TD.number_of_classes))
#
#     AISD = EOTrainingDatasetCollection().segmentationDataPipe(AISD_TD, path, crop=(512, 0.25), transform=transform)
#     dataloader = DataLoader2(AISD, batch_size=4, num_workers=4)
#     for batch in dataloader:
#         print(batch)
#
#
# def datasetsForCDTask():
#     AISD_TD = EOTrainingDatasetCollection()["AISD"]
#     print("Load training dataset: " + AISD_TD.name)
#     print("Number of training samples: " + str(AISD_TD.amount_of_training_data))
#     print("Number of classes: " + str(AISD_TD.number_of_classes))
#
#     AISD = EOTrainingDatasetCollection().segmentationDataPipe(AISD_TD, path, crop=(512, 0.25), transform=transform)
#     dataloader = DataLoader2(AISD, batch_size=4, num_workers=4)
#     for batch in dataloader:
#         print(batch)
#
# def tensorFlowSceneTask():
#     AID_tdml_encoding = EOTrainingDatasetCollection()['AID']
#     classes = ["Harbor"]
#
#     AID = EOTrainingDatasetCollection().sceneTensorDataset(AID_tdml_encoding, path).as_dataset()
#     iterator = AID.as_numpy_iterator()
#
#     # 遍历数据集
#
#     for i in range(500):
#         ite = next(iterator)
#
#         print(ite)
#
#
# def sceneDownloadTest():
#     dataset_library = EOTrainingDatasetCollection()
#     rs_sensetime = dataset_library["rs sensetime"]
#     # my_datasets = EOTrainingDatasetCollection().sceneDataset(rs_sensetime, path,
#     #                                                          download=True, transform=transform)
#     my_dataPipe = EOTrainingDatasetCollection().sceneDataPipe(rs_sensetime, path, transform)
#     dataloader = DataLoader2(my_dataPipe, batch_size=4, num_workers=4)
#
#     for batch in dataloader:
#         print(batch)
#
#
# def targetCrop():
#     dataset_library = EOTrainingDatasetCollection()
#     DOTA_2 = dataset_library["DOTA-v2.0"]
#     my_datasets = EOTrainingDatasetCollection().objectDataPipe(DOTA_2, ".", crop=(512, 0.25, 0.25), transform=transform)
#     dataloader = DataLoader2(my_datasets, batch_size=4, num_workers=4)
#
#     for batch in dataloader:
#         print(batch)
#
#
# def type_test():
#     merged_td_list = []
#     classes = []
#     eo = EOTrainingDataset(
#         id='whu_rs19',
#         name='WHU_RS19',
#         description="WHU-RS19 has 19 classes of remote sensing images scenes obtained from Google Earth",
#         tasks=[EOTask(task_type="Scene Classification",
#                       description="Structural high-resolution satellite image indexing")],
#         data=merged_td_list,
#         amount_of_training_data=len(merged_td_list),
#         classes=classes,
#         number_of_classes=len(classes),
#         bands=["red", "green", "blue"],
#         image_size="600x600"
#     )
#     print(eo)
#
#
# def acrossDataset():
#     AISD_TD = EOTrainingDatasetCollection()['RS-C11']
#     AID_tdml_encoding = EOTrainingDatasetCollection()['AID']
#     my_dataset = EOTrainingDatasetCollection().sceneAcrossDataset([AISD_TD, AID_tdml_encoding], ['Harbor'])
#     print(my_dataset)
#     myds = EOTrainingDatasetCollection().sceneTensorDataset(my_dataset, ".", transform)
#     dataloader = DataLoader2(myds, batch_size=4, num_workers=4)
#
#     for batch in dataloader:
#         print(batch)
#
#
# def acrossObjectDataset():
#     myds_td = EOTrainingDatasetCollection()["DOTA-v2.0"]
#     # myds = EOTrainingDatasetCollection().ObjectDataset(myds_td, ".", download=True, crop=(512, 0.25, 0.25), transform=target_transform)
#     myds = EOTrainingDatasetCollection().objectDataPipe(myds_td, path, crop=(512, 0.25, 0.25), transform=target_transform)
#     dataloader = DataLoader2(myds, batch_size=4, num_workers=4)
#
#     for batch in dataloader:
#         print(batch)
#
#
# data_path = r"D:\LiuSQi\Data"
#
#
# def acrossTensorObjectDataset():
#     myds_td = EOTrainingDatasetCollection()["DOTA-v2.0"]
#     # myds = EOTrainingDatasetCollection().ObjectDataset(myds_td, ".", download=True, crop=(512, 0.25, 0.25), transform=target_transform)
#     myds = EOTrainingDatasetCollection().objectTensorDataset(myds_td, data_path, crop=(512, 0.25, 0.25)).as_dataset()
#     for images, labels in myds:
#         # 在这里执行训练操作
#         print(type(images))
#         print(labels)
#
#
# def acrossSceneDataset():
#     myds_td = EOTrainingDatasetCollection()["rs sensetime"]
#     # myds = EOTrainingDatasetCollection().ObjectDataset(myds_td, ".", download=True, crop=(512, 0.25, 0.25), transform=target_transform)
#     myds = EOTrainingDatasetCollection().sceneTensorDataset(myds_td, path).as_dataset(4)
#     for images, labels in myds:
#         # 在这里执行训练操作
#         print(type(images))
#         print(labels)
#
#
# def acrossSegmentationDataset():
#     myds_td = EOTrainingDatasetCollection()["AISD"]
#     # myds = EOTrainingDatasetCollection().ObjectDataset(myds_td, ".", download=True, crop=(512, 0.25, 0.25), transform=target_transform)
#     myds = EOTrainingDatasetCollection().segmentationDataPipe(myds_td, ".", crop=(512, 0.25),
#                                                              transform=transform)
#     dataloader = DataLoader2(myds, batch_size=4, num_workers=4)
#
#     for batch in dataloader:
#         print(batch)
#
#
# def acrossCDDataset():
#     myds_td = EOTrainingDatasetCollection()["HRSCD"]
#     # myds = EOTrainingDatasetCollection().ObjectDataset(myds_td, ".", download=True, crop=(512, 0.25, 0.25), transform=target_transform)
#     myds = EOTrainingDatasetCollection().changeDetectionDatPipe(myds_td, ".", crop=(512, 0.25),
#                                                              transform=transform)
#     dataloader = DataLoader2(myds, batch_size=4, num_workers=4)
#
#     for batch in dataloader:
#         print(batch)
#
#
# def tensor_ori():
#
#     # Load the training dataset
#     training_dataset = read_from_json(r"D:\LiuSQi\Data\TDML-encoding\rs sensetime.json")  # read from TDML json file
#     # myds_td = EOTrainingDatasetCollection()["rs sensetime"]
#     # Transform the training dataset
#     class_map = pytdml.ml.create_class_map(training_dataset)  # create class map
#     train_dataset = pytdml.ml.TensorflowEOImageSceneTD(  # create TensorFlow train dataset
#         training_dataset.data,
#         class_map
#     )
#     tf_train_dataset = train_dataset.create_dataset()
#     for images, labels in tf_train_dataset:
#         # 在这里执行训练操作
#         print(type(images))
#         print(labels)
#
#
# def tdml_test():
#     # 获取目录资源中RS-C11数据集的 TDML 编码
#     tdml_encoding = EOTrainingDatasetCollection()['RSOD']
#
#     # 或者使用发布在网络上或本地的 TDML encoding 文件
#     # AISD = dataset_library.tdml_from_url("http://125.220.153.26/tdml/AISD.json")
#     # print("classes of AISD: " + str(AISD.classes))
#
#     # 获取RS-C11数据集的元数据信息
#     print("Load training dataset: " + tdml_encoding.name)
#     print("Number of training samples: " + str(tdml_encoding.amount_of_training_data))
#     print("Number of classes: " + str(tdml_encoding.number_of_classes))
#

def s3test():
    minio_client.fget_object("scene-classification", "//AID/Airport/Airport_version1_1.jpg", "./Airport_version1_9.jpg")
    # minio_client.fput_objct("scene-classification", "test.txt", "./Airport_version1_9.jpg",)

if __name__ == "__main__":
    # 下载数据和加载数据分离
    # datasetsForSceneTask()
    # 数据管道加载
    # datasetsForScenePipe()
    # 测试一
    # sceneDownloadTest()
    # 测试二
    # datasetsForSegmentationTask()
    # tdml_test()
    s3test()


