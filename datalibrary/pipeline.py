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
import time

from datalibrary.downloader import DatasetDownload, DatasetDownload2
from datalibrary.datasetcollection import Task
from pytdml import utils
from pytdml.ml import tdml_torch, tdml_torch_data_pipe, tdml_tensorflow
from pytdml.ml.ml_operators import create_classes_map_
from pytdml.type import EOTrainingDataset


class PipeLine:
    """
    dataset(EOTrainingDataset): EOTraining Dataset encoding of a dataset.
    root(str): Specify the download and working directory for the EOTraining dataset.
    """

    def __init__(self, dataset, root, crop=None):
        self.dataset = dataset
        self.root = root
        self.crop = crop

        if self.get_task() == Task.scene_classification:
            self.pipe = SceneClassificationTDPipeline(self.dataset, self.root)
        elif self.get_task() == Task.object_detection:
            self.pipe = ObjectDetectionTDPipeline(self.dataset, self.root, self.crop)
        elif self.get_task() == Task.semantic_segmentation:
            self.pipe = SemanticSegmentationTDPipeline(
                self.dataset, self.root, self.crop
            )
        elif self.get_task() == Task.change_detection:
            self.pipe = ChangeDetectionTDPipeline(self.dataset, self.root, self.crop)
        elif self.get_task() == Task.model_3d_reconstruction:
            self.pipe = Model3DReconstructionTDPipeLine(self.dataset, self.root)
        else:
            raise ValueError(
                "Dataset task type parsing error! Please recheck your input parameters."
            )

    def get_task(self):
        return self.dataset.tasks[0].task_type

    def torch_dataset(self, download=True, transform=None):
        return self.pipe.torch_dataset(download, transform)

    def torch_data_pipe(self, transform=None):
        return self.pipe.torch_data_pipe(transform)

    def tensorflow_data_pipe(self):
        return self.pipe.tensorflow_data_pipe()


class SceneClassificationTDPipeline:

    def __init__(self, dataset, root):
        self.dataset = dataset
        self.root = root

    def torch_dataset(
        self, download=True, transform=None
    ) -> tdml_torch.TorchSceneClassificationTD:
        """
        Args:
            download(bool, optional): Whether to download the dataset or not.
            transform(Transform, optional):A function to apply transformations to the data. Defaults to None.

        Returns:
            Torch Dataset for EO image scene classification training dataset.
        """
        cls_list = self.dataset.classes
        class_map = create_classes_map_(cls_list)

        cache_file_path = utils.generate_cache_file_path(self.root, self.dataset.name)
        td_list = utils.load_cached_training_data(cache_file_path)
        # if the dataset is initially local
        if len(td_list) == 0:
            td_list = self.dataset.data

        if download:
            import time

            # 记录开始时间
            # start_time = time.time()
            td_list = DatasetDownload(
                Task.scene_classification, self.dataset, self.root
            )
            # 记录结束时间
            # end_time = time.time()

            # 计算并打印执行时间
            # print(f"Code block execution time: {end_time - start_time} seconds")
            utils.cache_dump(cache_file_path, self.root, td_list)

        if len(td_list) == 0:
            raise ValueError(
                "No dataset found in the directory {}, "
                "please make sure the dataset has been downloaded in it".format(
                    self.root
                )
            )

        return tdml_torch.TorchSceneClassificationTD(
            td_list, self.root, class_map, transform
        )

    def torch_data_pipe(self, transform=None):
        """
         Returns a data pipe for training or inference on a scene dataset.
        Args:
            transform (Optional[Callable]): An optional function to transform the input and target images.
            Defaults to None.
        Returns:
            TorchDPEOImageSceneTD: A data pipe object that can be used for training or inference on a scene dataset.
        """
        # if isinstance(self.dataset, EOTrainingDataset):
        class_map = create_classes_map_(self.dataset.classes)
        cache_path = utils.generate_cache_file_path(self.root, self.dataset.name)
        return tdml_torch_data_pipe.TorchSceneClassificationDataPipe(
            self.dataset.data, self.root, cache_path, class_map, transform
        )

    def tensorflow_data_pipe(self):

        class_map = create_classes_map_(self.dataset.classes)
        cache_path = utils.generate_cache_file_path(self.root, self.dataset.name)
        return tdml_tensorflow.TensorSceneClassificationDataPipe(
            self.dataset, self.root, cache_path, class_map
        )


class ObjectDetectionTDPipeline:
    def __init__(self, dataset, root, crop=None):
        """
        crop(tuple or list, optional): (crop_size, overlap, thread).
                crop_size (int): The size of the crop in pixels.
                overlap (float): The overlap ratio between two adjacent crops, ranging from 0 to 1.
                threshold (float): The threshold for the minimum ratio of target area to crop area to be considered
                as a valid target.
        """
        self.dataset = dataset
        self.root = root
        self.crop = crop

    def torch_dataset(
        self, download=True, transform=None
    ) -> tdml_torch.TorchObjectDetectionTD:
        """
        Args:

            download(bool, optional): Whether to download the dataset or not.
            transform(Transform, optional):A function to apply transformations to the data. Defaults to None.

        Returns:
            Torch Dataset for EO image object detection training dataset.
        """

        class_map = create_classes_map_(self.dataset.classes)

        cache_file_path = utils.generate_cache_file_path(
            self.root, self.dataset.name, self.crop
        )
        td_list = utils.load_cached_training_data(cache_file_path)
        # if the dataset is initially local
        if len(td_list) == 0:
            td_list = self.dataset.data
        if download:
            start = time.time()
            td_list = DatasetDownload2(
                Task.object_detection, self.dataset, self.root, self.crop
            )
            end = time.time()
            print("total download time: " + str(end - start))
            utils.cache_dump(cache_file_path, self.root, td_list)
        if len(td_list) == 0:
            raise ValueError(
                "No dataset found in the directory {}, "
                "please make sure the dataset has been downloaded in it".format(
                    self.root
                )
            )
        return tdml_torch.TorchObjectDetectionTD(
            td_list, self.root, class_map, transform
        )

    def tensorflow_data_pipe(self):
        class_map = create_classes_map_(self.dataset.classes)

        return tdml_tensorflow.TensorObjectDetectionDataPipe(
            self.dataset, self.root, class_map, self.crop
        )

    def torch_data_pipe(
        self, transform=None
    ) -> tdml_torch_data_pipe.TorchObjectDetectionDataPipe:
        """
        Returns a data pipe for training an object detection model using the given dataset.
        Args:

            transform (Optional[Callable]): An optional function to transform the input and target images. Defaults to None.
        Returns:
            TorchDPEOImageObjectTD: A data pipe object that can be used for training an object detection model.
        """
        class_map = create_classes_map_(self.dataset.classes)
        cache_path = utils.generate_cache_file_path(
            self.root, self.dataset.name, self.crop
        )
        return tdml_torch_data_pipe.TorchObjectDetectionDataPipe(
            self.dataset.data, self.root, cache_path, class_map, self.crop, transform
        )


class SemanticSegmentationTDPipeline:

    def __init__(self, dataset, root, crop=None):
        """
        crop (Optional[Tuple[int, int]]): An optional tuple of two integers representing the size of the cropped
                 images. Defaults to None.
        """
        self.dataset = dataset
        self.root = root
        self.crop = crop

    def torch_data_pipe(self, transform=None):
        """
        Returns a data pipe for training an image segmentation model using the given dataset.
        Args:

            transform (Optional[Callable]): An optional function to transform the input and target images. Defaults
            to None.
        Returns:
            TorchDPEOImageSegmentationTD: A data pipe object that can be used for training an image segmentation model.
        Raises:
            ValueError: If the dataset classes are not provided.
        Notes:
            This method generates a cache file in the root directory to speed up data loading.
            The cache file is generated using the `get_cache_path` function from the `eo_utils` module.
        """

        cache_path = utils.generate_cache_file_path(
            self.root, self.dataset.name, self.crop
        )
        return tdml_torch_data_pipe.TorchSemanticSegmentationDataPipe(
            self.dataset.data,
            self.root,
            cache_path,
            self.dataset.classes,
            self.crop,
            transform,
        )

    def tensorflow_data_pipe(self, training_type):
        cls_list = self.dataset.classes
        return tdml_tensorflow.TensorSemanticSegmentationDataPipe(
            self.dataset, self.root, training_type, cls_list, self.crop
        )

    def torch_dataset(
        self, download=True, transform=None
    ) -> tdml_torch.TorchSemanticSegmentationTD:
        """
        Args:

            download(bool, optional): Whether to download the dataset or not.
            transform(Transform, optional):A function to apply transformations to the data. Defaults to None.

        Returns:
            Torch Dataset for EO image object detection training dataset.
        """
        # if isinstance(self.dataset, EOTrainingDataset):

        cache_file_path = utils.generate_cache_file_path(
            self.root, self.dataset.name, self.crop
        )
        td_list = utils.load_cached_training_data(cache_file_path)
        # if the dataset is initially local
        if len(td_list) == 0:
            td_list = self.dataset.data
        if download:

            td_list = DatasetDownload2(
                Task.semantic_segmentation, self.dataset, self.root, self.crop
            )

            utils.cache_dump(cache_file_path, self.root, td_list)
        return tdml_torch.TorchSemanticSegmentationTD(
            td_list, self.root, self.dataset.classes, transform
        )


class ChangeDetectionTDPipeline:

    def __init__(self, dataset, root, crop=None):
        self.dataset = dataset
        self.root = root
        self.crop = crop

    def torch_dataset(self, download=True, transform=None):
        """
        Returns a change detection dataset for training or inference.

        Args:

            download (bool): Whether or not to download the dataset from a remote server. Defaults to True.
            transform (Optional[Callable]): An optional function to transform the input and target data. Defaults to
            None.
        Returns:
            List[tdml_torch.TorchEOImageChangeDetectionTD]: A list of TorchEOImageChangeDetectionTD objects that can
            be used for training or inference.
        """

        if isinstance(self.dataset, EOTrainingDataset):

            cache_file_path = utils.generate_cache_file_path(
                self.root, self.dataset.name, self.crop
            )
            td_list = utils.load_cached_training_data(cache_file_path)
            # if the dataset is initially local
            if len(td_list) == 0:
                td_list = self.dataset.data
            if download:
                td_list = DatasetDownload2(
                    Task.change_detection, self.dataset, self.root, self.crop
                )
                utils.cache_dump(cache_file_path, self.root, td_list)

            return tdml_torch.TorchChangeDetectionTD(td_list, self.root, transform)

    def torch_data_pipe(self, transform=None):
        """
        Returns a data pipe for training an image change detection model using the given dataset.

        Args:
            transform (Optional[Callable]): An optional function to transform the input and target images. Defaults
            to None.

        Returns:
            TorchDPEOImageChangeDetectionTD: A data pipe object that can be used for training an image change
             detection model.
        """
        cache_path = utils.generate_cache_file_path(
            self.root, self.dataset.name, self.crop
        )
        return tdml_torch_data_pipe.TorchChangeDetectionDataPipe(
            self.dataset.data, self.root, cache_path, self.crop, transform
        )

    def tensorflow_data_pipe(self):
        raise NotImplementedError("This function is not yet implemented")


# class StereoDataset(AIDatasetLoaderInTask):
#     def __init__(self, dataset, root):
#         super().__init__(dataset, root)


class Model3DReconstructionTDPipeLine:
    def __init__(self, dataset, root):
        self.dataset = dataset
        self.root = root

    def torch_dataset(self, download, transform):
        """
        Returns a stereo image dataset for training or inference.

        Args:

            download (bool): Whether or not to download the dataset from a remote server.
            transform (Optional[Callable]): An optional function to transform the input and target images. Defaults to None.

        Returns:
            TorchEOImageStereoTD: A stereo image dataset object that can be used for training or inference.
        """

        cache_file_path = utils.generate_cache_file_path(
            self.root, self.dataset.name, self.crop
        )
        td_list = utils.load_cached_training_data(cache_file_path)
        # if the dataset is initially local
        if len(td_list) == 0:
            td_list = self.dataset.data
        if download:
            td_list = DatasetDownload(
                Task.model_3d_reconstruction, self.dataset, self.root
            )
            utils.cache_dump(cache_file_path, self.root, td_list)
        return tdml_torch.TorchStereoTD(td_list, self.root, transform)

    def torch_data_pipe(self, transform=None):
        return tdml_torch_data_pipe.TorchStereoDataPipe(
            self.dataset.data, self.root, transform
        )

    def tensorflow_data_pipe(self):
        raise NotImplementedError("This function is not yet implemented")
