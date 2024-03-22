# pytdml

[pytdml](https://github.com/openrsgis/pytdml) is a pure python parser and encoder for training datasets based on OGC
Training Data Markup Language for AI standard.

---

## Installation

The package can be installed via pip.

### Requirements

* Python 3 and above

### Dependencies

Dependencies are listed in [requirements.txt](https://github.com/openrsgis/pytdml/blob/main/requirements.txt). Dependencies are automatically installed during
pytdml's installation.

### Installing the Package

```bash
pip install pytdml
```

---

## Usage

### Encoding

#### From the command line

The training dataset can be encoded to TrainingDML-AI JSON format by YAML configuration file with command line.

```bash
pytdml/yaml_to_tdml.py --config=<YAML configuration file path> --output=<Output TrainingDML-AI JSON file path>
```

YAML configuration file schema is described in [encoding YAML configuration file schema](https://github.com/openrsgis/pytdml/blob/main/encoding_config_schema.yml).

#### Using the API from python

The training dataset can also be encoded to TrainingDML-AI JSON format with Python API.

```python
from pytdml.io import write_to_json
from pytdml.type import EOTrainingDataset, EOTrainingData, EOTask, EODataSource, SceneLabel

# generate EO training dataset
dataset = EOTrainingDataset(
    id='...',
    name='...',
    description='...',
    data=[
        EOTrainingData(
            id='...',
            labels=[
                SceneLabel(
                    label_class='...',
                    data_url='...',
                    date_time='...'),
                ...
            ]),
        ...
    ],
    version="...",
    amount_of_training_data=...,
    created_time="...",
    updated_time="...",
    providers=["..."],
    keywords=["...", "..."],
    tasks=[EOTask(task_type="...",
                  description="...")],
    data_sources=[EODataSource(
        id="...",
        data_type="...",
        resolution="..."
    )],
    classes=["...", "...", "..."],
    number_of_classes=...,
    bands=["...", "...", "..."],
    image_size="..."
)
# write to json
write_to_json(dataset, "dataset.json")
```

#### Encoding training data from S3

```python
# get training data from s3

s3_client = pytdml.datalibrary.S3Client('s3', "your_server", "your_ak", "your_sk")

td_list = []
bucket_name = "my-bucket"
obj_list = s3_client.list_objects(Bucket=bucket_name, Prefix="whu_rs19/")
for obj in obj_list:
    td = EOTrainingData(
        id=obj.split(".")[0],
        labels=[SceneLabel(label_class=obj.split("/")[1])],
        data_url=f"s3://{bucket_name}/{quote(obj)}",
        date_time="2010"
    )
    td_list.append(td)
```

### Parsing

The training dataset described with TrainingDML-AI JSON file can be parsed with python API and transformed to
PyTorch/TensorFlow dataset.

#### Read TrainingDataset object from JSON file

```python
import pytdml

training_dataset = pytdml.io.read_from_json("dataset.json")  # read from TDML json file
print("Load training dataset: " + training_dataset.name)
print("Number of training samples: " + str(training_dataset.amount_of_training_data))
print("Number of classes: " + str(training_dataset.number_of_classes))
```

#### Read training data from s3

```python
import pytdml

# Initialize S3client 
s3_client = pytdml.io.S3_reader.S3Client("s3", "your_server", "your_akey", "your_skey")
# Load the training dataset
training_dataset = pytdml.io.read_from_json("dataset.json")  # read from TDML json file
for item in training_dataset.data:
    path = item.data_url
    if pytdml.io.S3_reader.pasrse_s3_path(path):
        bucket_name, key_name = pytdml.io.S3_reader.parse_s3_path(path)
        object_data = s3_client.get_object(bucket_name, key_name)
        # Process the S3 object data (read as PIL Image)
        with PIL.Image.open(BytesIO(object_data)) as img:
            # processing....
    else:
        print("Invalid S3 path:", path)
```

#### Transform to PyTorch dataset

* Scene classification dataset

```python
import pytdml
from torchvision import transforms

# Load the training dataset
training_dataset = pytdml.io.read_from_json("dataset.json")  # read from TDML json file

# Transform the training dataset
class_map = pytdml.ml.creat_class_map(training_dataset)  # create class map
train_dataset = pytdml.ml.TorchEOImageSceneTD(  # create Torch train dataset
    training_dataset.data,
    class_map,
    transform=transforms.Compose(  # transform for the training set
        [transforms.RandomResizedCrop(size=156, scale=(0.8, 1.0)),  # random resize
         transforms.RandomRotation(degrees=15),  # random rotate
         transforms.RandomHorizontalFlip(),  # random flip
         transforms.CenterCrop(size=124),  # center crop
         transforms.ToTensor(),  # transform to tensor
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalize
         ]
    ))
```

* Object detection dataset

```python
import pytdml

# Load the training dataset
training_dataset = pytdml.io.read_from_json("dataset.json")  # read from TDML json file

# Transform the training dataset
class_map = pytdml.ml.creat_class_map(training_dataset)  # create class map
train_dataset = pytdml.ml.TorchEOImageObjectTD(  # create Torch train dataset
    training_dataset.data,
    class_map,
    transform=pytdml.ml.BaseTransform([128, 128])
)
```

* Semantic segmentation dataset

```python
import pytdml
from torchvision import transforms

# Load the training dataset
training_dataset = pytdml.io.read_from_json("dataset.json")  # read from TDML json file

# Transform the training dataset
class_map = pytdml.ml.creat_class_map(training_dataset)  # create class map
train_dataset = pytdml.ml.TorchEOImageSegmentationTD(  # create Torch train dataset
    training_dataset.data,
    class_map,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
)
```

#### Transform to TensorFlow dataset

* Scene classification dataset

```python
import pytdml

# Load the training dataset
training_dataset = pytdml.io.read_from_json("dataset.json")  # read from TDML json file

# Transform the training dataset
class_map = pytdml.ml.creat_class_map(training_dataset)  # create class map
train_dataset = pytdml.ml.TensorflowEOImageSceneTD(  # create TensorFlow train dataset
    training_dataset.data,
    class_map
)
tf_train_dataset = train_dataset.create_dataset()
```

* Object detection dataset

```python
import pytdml

# Load the training dataset
training_dataset = pytdml.io.read_from_json("dataset.json")  # read from TDML json file

# Transform the training dataset
class_map = pytdml.ml.creat_class_map(training_dataset)  # create class map
train_dataset = pytdml.ml.TensorflowEOImageObjectTD(  # create TensorFlow train dataset
    training_dataset.data,
    class_map
)
tf_train_dataset = train_dataset.create_dataset()
```

* Semantic segmentation dataset

```python
import pytdml

# Load the training dataset
training_dataset = pytdml.io.read_from_json("dataset.json")  # read from TDML json file

# Transform the training dataset
class_map = pytdml.ml.creat_class_map(training_dataset)  # create class map
train_dataset = pytdml.ml.TensorflowEOImageSegmentationTD(  # create TensorFlow train dataset
    training_dataset.data,
    class_map
)
tf_train_dataset = train_dataset.create_dataset()
```

### Image Cropping

The images of training dataset in TrainingDML-AI JSON format can be cropped with command line for preprocessing.

```bash
pytdml/tdml_image_crop.py  --input=<Input original TrainingDML-AU file path> --output_json=<Output result TrainingDML-AI JSON file path>
                          --output_images=<Output dir of result cropped images> --size=<Crop size of images>
```


###　Load datasets to provide training-ready training data

- View available datasets according to different tasks and categories

```python
ds_lib = EOTrainingDatasetCollection()

# Find the EO dataset that contains the category 'Building Area' in the Semantic Segmentation task in the Server.
ds_lib.dataset_list(Task.semantic_segmentation, ["Building Area"]) 

```

- Load the TrainingDML code for the corresponding dataset

```python
aisd_tds = ds_lib["AISD"]
# Output to obtain metadata information such as the name of the corresponding dataset, the number of training samples, the number of categories included in the dataset, etc.
print("Load training dataset: " + aisd_tds.name)
print("Number of training samples: " + str(aisd_tds.amount_of_training_data))
print("Number of classes: " + str(aisd_tds.number_of_classes))
```

- Acquire more training datas of this category across datasets

```python
building_extraction_td = ds_lib["buildingExtraction"]
# Combine training datas from two datasets with the category 'Building Area' into one training data collection
my_dataset_td = ds_lib.training_data_collection(Task.semantic_segmentation, [aisd_td, building_extraction_td], ["Building Area"])
```

- Call the pipeline framework's wrapper class to encapsulate the training datas into a trainable dataset

```python
transform = transforms.Compose(  # transform for the dataset
    [
        transforms.ToTensor(),
        transforms.CenterCrop(224),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),  # random flip
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalize
    ]
)
# transforms for object detection task
import pytdml.ml.object_transforms as transform_target
target_transform = transform_target.Compose([
    transform_target.ToTensor(),
    transform_target.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transform_target.RandomResize((512, 512))
])
# Split the dataset by a customized ratio
train_data, val_data, test_data = pytdml.ml.split_data(my_dataset_td.data, 0.7, 0.2, 0.1)
# OR split the dataset by training type of training data 
# train_data = pytdml.ml.split_data(my_dataset_td.data, "train")
# valid_data = pytdml.ml.split_data(my_dataset_td.data, "valid")
path = "." 
# Wrapping dataset classes with pytorch framework
train_set = PipeLine(train_data, path).torch_dataset(download=True, transform=transform)
valid_set = PipeLine(valid_data, path).torch_dataset(download=True, transform=transform)
# dataPipe load
# train_set = PipeLine(train_data, path).torch_dataPipe(transform=transform)

train_dataloader = DataLoader(train_set, batch_size=4, num_workers=4)
val_dataloader = DataLoader(val_set, batch_size=4, num_workers=4)
# Create the model
net = DeepLabV3()
criterion = nn.NLLLoss()
params = add_weight_decay(net, l2_value=0.0001)
optimizer = torch.optim.Adam(params, lr=1e-3)
# Train the network
for e in range(100):
    print("Epoch: " + str(e))
    net = net.train()
    train_loss = 0
    train_acc = 0
    train_acc_cls = 0
    train_mean_iu = 0
    train_fwavacc = 0
    prev_time = datetime.datetime.now()
    for iter_i, data in enumerate(train_dataloader):
        # forward
        # ...
        # backword
        # ...
        with torch.no_grad():
            for data in val_dataloader:
                # ...
```

### Convert other EO dataset formats to TrainingDML-AI format

- convert coco format to TrainingDMl-AI format:

```python
from pytdml.convert_utils import convert_coco_to_tdml,convert_stac_to_tdml

coco_path = "/mnt/example/coco_file.json"
output_path = "convert_result.json"

convert_coco_to_tdml(coco_path, output_path)
```

