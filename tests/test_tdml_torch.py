import pytest
import pytdml
import pytdml.io
import pytdml.ml
from torchvision import transforms


def test_torch_eo_image_object_td():
    # Load the training dataset
    training_dataset = pytdml.io.read_from_json(
        "tests/data/object-detection/COWC_partial.json"
    )  # read from TDML json file
    print("Load training dataset: " + training_dataset.name)
    print(
        "Number of training samples: " + str(training_dataset.amount_of_training_data)
    )
    print("Number of classes: " + str(training_dataset.number_of_classes))

    # Set parameters
    train_size = [128, 128]

    # Prepare the training dataset
    class_map = pytdml.ml.create_class_map(training_dataset)  # create class map
    train_dataset = pytdml.ml.TorchEOImageObjectTD(  # create Torch train dataset
        training_dataset.data, class_map, transform=pytdml.ml.BaseTransform(train_size)
    )

    img, label, img_height, img_width = train_dataset[0]


def test_torch_eo_image_scene_td():
    training_dataset = pytdml.io.read_from_json(
        "tests/data/scene-classification/WHU-RS19.json"
    )  # read from TDML json file
    print("Load training dataset: " + training_dataset.name)
    print(
        "Number of training samples: " + str(training_dataset.amount_of_training_data)
    )
    print("Number of classes: " + str(training_dataset.number_of_classes))

    # Prepare the training dataset
    class_map = pytdml.ml.create_class_map(training_dataset)  # create class map
    trans_size = [64, 64]
    train_dataset = pytdml.ml.TorchEOImageSceneTD(  # create Torch train dataset
        training_dataset.data,
        class_map,
        transform=transforms.Compose(  # transform for the training set
            [
                transforms.RandomResizedCrop(
                    size=156, scale=(0.8, 1.0)
                ),  # random resize
                transforms.RandomRotation(degrees=15),  # random rotate
                transforms.RandomHorizontalFlip(),  # random flip
                transforms.CenterCrop(size=124),  # center crop
                transforms.ToTensor(),  # transform to tensor
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),  # normalize
            ]
        ),
    )

    img, label = train_dataset[0]


def test_torch_eo_image_segmentation_td():
    # Load the training dataset
    training_dataset = pytdml.io.read_from_json(
        "tests/data/semantic_segmentation/GID-5C.json"
    )  # read from TDML json file
    print("Load training dataset: " + training_dataset.name)
    print(
        "Number of training samples: " + str(training_dataset.amount_of_training_data)
    )
    print("Number of classes: " + str(training_dataset.number_of_classes))

    # Prepare the training dataset
    class_map = pytdml.ml.create_class_map(training_dataset)  # create class map
    train_set, val_set, test_set = pytdml.ml.split_train_valid_test(
        training_dataset, 0.7, 0.2, 0.1
    )  # split dataset
    train_dataset = pytdml.ml.TorchEOImageSegmentationTD(  # create Torch train dataset
        train_set,
        class_map,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )

    val_dataset = pytdml.ml.TorchEOImageSegmentationTD(
        val_set,
        class_map,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )
