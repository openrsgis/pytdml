from build.lib.pytdml.type.basic_types import NamedValue

# pytdml

[pytdml](https://github.com/openrsgis/pytdml) is a pure python parser and encoder for training datasets based on OGC
Training Data Markup Language for AI standard.

---
## Installation

> **⚠️ Current Package Status**  
> 
> The PyPI repository currently hosts an outdated version of PyTDML.
> As we are actively updating and maintaining the library to ensure full functionality, 
> **direct installation via `pip install` is temporarily unavailable**.
> 
> We provide pre-built artifacts for immediate use:

### Pre-built Artifacts
The repository includes the following files for direct installation:
- `pytdml-1.2.0-py3-none-any.whl`: Pre-built wheel package for Python 3
- `Dockerfile`: Docker configuration for containerized deployment

> **Functionality Scope**  
> The current version (1.2.0) provides a base and light installation, including core functionality:
> - Full implementation of TrainingDML-AI class definitions
> - Dataset parsing and encoding capabilities (IO functionality)
> 
> ⚠️ **Machine Learning (ML) specific features are under active development and update** and will be included in future releases.

### Requirements

* **For wheel installation**: 
  - Python 3.9 or 3.10
  - [pip](https://pip.pypa.io/en/stable/installation/) package manager

* **For Docker deployment**:
  - [Docker](https://docs.docker.com/get-docker/) installed and configured

#### Method 1: Install from GitHub Release
Install the pre-built wheel package directly from GitHub Releases:

Option A: Direct URL Installation

   ```bash
   # Base installation (core functionality)
   pip install https://github.com/openrsgis/pytdml/releases/download/v1.2.0/pytdml-1.2.0-py3-none-any.whl
    
   # With IO functionality After base installation (additional dependencies)
   pip install pytdml[io]
   ```

Option B: Local Installation
1. **Download the .whl file**: Navigate to Releases page and download pytdml-1.2.0-py3-none-any.whl from the **Assets** section.
   
2. **Install the wheel package**:
   ```bash
   # Base installation
   pip install pytdml-1.2.0-py3-none-any.whl
   
   # IO additional dependencies(The functionality under `pytdml.io` requires additional packages for handling different formats and network communications.)
   pip install pytdml-1.2.0-py3-none-any.whl[io]
   ```
   
#### Method 2: Docker Container Deployment
1. **Build the Docker image from the provided Dockerfile**:
   ```bash
   docker build -t pytdml-base:1.2.0 .
   ```
   
2. **Run Python with PyTDML in a container**:   
   
   - Interactive mode:
   ```bash
   docker run -it --rm --name pytdml-python pytdml-base:1.2.0 python
   ```

   - Execute a script:
   ```bash
   docker run -it --rm -v "$(pwd)":/workspace pytdml-base:1.2.0 python /workspace/your_script.py
   ```
---

## Usage

### Encoding

#### 1. From the command line

The training dataset can be encoded to TrainingDML-AI JSON format by YAML configuration file with command line.

```bash
python -m pytdml.io.yaml_converter.py --config=<YAML configuration file path> --output=<Output TrainingDML-AI JSON file path>
```

YAML configuration file schema is described in [encoding YAML configuration file schema](https://github.com/openrsgis/pytdml/blob/main/encoding_config_schema.yml).

#### 2. Using the API from python

The training dataset can also be encoded to TrainingDML-AI JSON format with Python API.

```python
from pytdml.type import EOTrainingDataset, AI_EOTask, AI_EOTrainingData, AI_SceneLabel, MD_Band, MD_Identifier, NamedValue, CI_Citation, MD_Scope, AI_Labeling, AI_MetricsInLiterature, DataQuality, AI_TDChangeset

# Generate EO training dataset with required and optional fields
dataset = EOTrainingDataset(
    # required fields
    id='...',
    name='...',
    description='...',
    license='...',
    type='AI_EOTrainingDataset',

    # Task definition (at least one required)
    tasks=[
        AI_EOTask(
            id='...',
            task_type='...',
            type='AI_EOTask'
        ),
        ...
    ],

    # Training data (at least one required)
    data=[
        AI_EOTrainingData(
            id='...',
            data_url=['...'],
            labels=[
                AI_SceneLabel(
                    label_class='...',
                    type='AI_SceneLabel'
                ),
               ...
            ],
            type="AI_EOTrainingData"
        ),
        ...
    ],

    # Optional fields
    bands=[
        MD_Band(
           name=[
              MD_Identifier(
                 code='...'
              )
           ]
        ),
       ...
    ],
    extent=[...],
    image_size='...',
    amount_of_training_data=...,
    classes=[
        NamedValue(
            key='...',
            value=...
        ),
        ...
    ],
    classification_schema='...',
    created_time='...',
    data_sources=[
        CI_Citation(
            title='...'
        ),
       ...
    ],
    doi='...',
    keywords=['...', ...],
    number_of_classes=...,
    providers=['...', ...],
    scope=MD_Scope(
        level='...'
    ),
    statistics_info=[
        NamedValue(
            key='...',
            value=...
        ),
        ...
    ],
    updated_time='...',
    version='...',
    labeling=[
        AI_Labeling(
            id='...',
            scope=MD_Scope(
                level='...'
            ),
            type='AI_Labeling'
        ),
        ...
    ],
    metrics_in_LIT=[
        AI_MetricsInLiterature(
            doi='...',
            metrics=[
                NamedValue(
                    key='...',
                    value=...
                ),
                ...
            ]
        ),
       ...
    ],
    quality=[
        DataQuality(
            type='DataQuality',
            scope=MD_Scope(
                level='...'
            )
        ),
       ...
    ],
    changesets=[
        AI_TDChangeset(
            type='AI_TDChangeset',
            id='...',
            change_count=...
        ),
        ...
    ]
)

# Write to JSON file
from pytdml.io import write_to_json

write_to_json(dataset, "eo_dataset.json")
```

### Parsing

The training dataset described with TrainingDML-AI JSON file can be parsed with python API and transformed to
PyTorch/TensorFlow dataset.

#### Read TrainingDataset object from JSON file

```python
import pytdml.io

training_dataset = pytdml.io.read_from_json("dataset.json")  # read from TDML json file
print("Load training dataset: " + training_dataset.name)
print("Number of training samples: " + str(training_dataset.amount_of_training_data))
print("Number of classes: " + str(training_dataset.number_of_classes))
```

#### Transform to PyTorch dataset

* Scene classification dataset

```python
import pytdml.ml
from torchvision import transforms

# Load the training dataset
training_dataset = pytdml.io.read_from_json("dataset.json")  # read from TDML json file

# Transform the training dataset
class_map = pytdml.ml.create_class_map(training_dataset)  # create class map
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
import pytdml.ml

# Load the training dataset
training_dataset = pytdml.io.read_from_json("dataset.json")  # read from TDML json file

# Transform the training dataset
class_map = pytdml.ml.create_class_map(training_dataset)  # create class map
train_dataset = pytdml.ml.TorchEOImageObjectTD(  # create Torch train dataset
    training_dataset.data,
    class_map,
    transform=pytdml.ml.BaseTransform([128, 128])
)
```

* Semantic segmentation dataset

```python
import pytdml.ml
from torchvision import transforms

# Load the training dataset
training_dataset = pytdml.io.read_from_json("dataset.json")  # read from TDML json file

# Transform the training dataset
class_map = pytdml.ml.create_class_map(training_dataset)  # create class map
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
import pytdml.ml

# Load the training dataset
training_dataset = pytdml.io.read_from_json("dataset.json")  # read from TDML json file

# Transform the training dataset
class_map = pytdml.ml.create_class_map(training_dataset)  # create class map
train_dataset = pytdml.ml.TensorflowEOImageSceneTD(  # create TensorFlow train dataset
    training_dataset.data,
    class_map
)
tf_train_dataset = train_dataset.create_dataset()
```

* Object detection dataset

```python
import pytdml.ml

# Load the training dataset
training_dataset = pytdml.io.read_from_json("dataset.json")  # read from TDML json file

# Transform the training dataset
class_map = pytdml.ml.create_class_map(training_dataset)  # create class map
train_dataset = pytdml.ml.TensorflowEOImageObjectTD(  # create TensorFlow train dataset
    training_dataset.data,
    class_map
)
tf_train_dataset = train_dataset.create_dataset()
```

* Semantic segmentation dataset

```python
import pytdml.ml

# Load the training dataset
training_dataset = pytdml.io.read_from_json("dataset.json")  # read from TDML json file

# Transform the training dataset
class_map = pytdml.ml.create_class_map(training_dataset)  # create class map
train_dataset = pytdml.ml.TensorflowEOImageSegmentationTD(  # create TensorFlow train dataset
    training_dataset.data,
    class_map
)
tf_train_dataset = train_dataset.create_dataset()
```

