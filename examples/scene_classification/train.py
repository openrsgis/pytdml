import numpy as np
from torch import nn, optim
from torchvision import transforms

from model.train_and_valid import train_and_valid
from model.resnet import ResNet18_OS8
from utils.summaries import TensorboardSummary

import pytdml
import matplotlib.pyplot as plt

# Load the training dataset


training_dataset = pytdml.read_from_json("whu_rs19.json")  # read from TDML json file
print("Load training dataset: " + training_dataset.name)
print("Number of training samples: " + str(training_dataset.amount_of_training_data))
print("Number of classes: " + str(training_dataset.number_of_classes))

# Prepare the training dataset
class_map = pytdml.creat_class_map(training_dataset)  # create class map
train_set, val_set, test_set = pytdml.split_train_valid_test(training_dataset, 0.7, 0.2, 0.1)  # split dataset
train_dataset = pytdml.TorchEOImageSceneTD(  # create Torch train dataset
    test_set,
    class_map,
    transforms.Compose(  # transform for the training set
        [transforms.RandomResizedCrop(size=156, scale=(0.8, 1.0)),  # random resize
         transforms.RandomRotation(degrees=15),  # random rotate
         transforms.RandomHorizontalFlip(),  # random flip
         transforms.CenterCrop(size=124),  # center crop
         transforms.ToTensor(),  # transform to tensor
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # normalize
         ]
    ))
train_transforms = test_valid_transforms = transforms.Compose(  # transform for the test set
    [transforms.Resize(156),
     transforms.CenterCrop(124),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
     ]
)
val_dataset = pytdml.TorchEOImageSceneTD(val_set, class_map, test_valid_transforms)  # create Torch val dataset
test_dataset = pytdml.TorchEOImageSceneTD(test_set, class_map, test_valid_transforms)  # create Torch test dataset

# Create model
resnet50 = ResNet18_OS8()  # create ResNet50 model
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet50.parameters(), lr=0.01)

# Train model
num_epochs = 1000
batch_size = 128
save_path = "result"
summary = TensorboardSummary(directory=save_path)
writer = summary.creater_summary()
trained_model, record = train_and_valid(resnet50, train_dataset, val_dataset, train_dataset.size, val_dataset.size,
                                        writer, loss_func, optimizer, num_epochs, batch_size)

# plot the training loss and accuracy
record = np.array(record)
plt.plot(record[:, 0:2])
plt.legend(['Train Loss', 'Valid Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(0, 1)
plt.savefig('loss.png')
plt.show()
plt.plot(record[:, 2:4])
plt.legend(['Train Accuracy', 'Valid Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.savefig('accuracy.png')
plt.show()
