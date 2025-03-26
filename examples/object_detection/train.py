import pytdml
import torch
import torch.optim as optim
from model.utils import tools
from model.yolo import myYOLO
from torch.utils.data import DataLoader

# Load the training dataset
training_dataset = pytdml.io.read_from_json("dota_v1.json")  # read from TDML json file
print("Load training dataset: " + training_dataset.name)
print("Number of training samples: " + str(training_dataset.amount_of_training_data))
print("Number of classes: " + str(training_dataset.number_of_classes))

# Set parameters
train_size = [128, 128]
device = torch.device("cpu")
batch_size = 16
max_epoch = 10
lr_epoch = (60, 90, 160)

# Prepare the training dataset
class_map = pytdml.ml.creat_class_map(training_dataset)  # create class map
train_dataset = pytdml.ml.TorchEOImageObjectTD(  # create Torch train dataset
    training_dataset.data, class_map, transform=pytdml.ml.BaseTransform(train_size)
)
dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=tools.detection_collate,
    num_workers=0,
    pin_memory=True,
)

# Create model
yolo_net = myYOLO(
    device,
    input_size=train_size,
    num_classes=training_dataset.number_of_classes,
    trainable=True,
)
model = yolo_net
model.to(device).train()
print("Let us train yolo net!")

# Create Optimizer
base_lr = 1e-5
tmp_lr = base_lr
optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=5e-4)
epoch_size = len(train_dataset) // batch_size

# Start training
for epoch_ in range(0, max_epoch):
    print("Epoch: %d" % epoch_)
    if epoch_ in lr_epoch:
        tmp_lr = tmp_lr * 0.1
        tools.set_lr(optimizer, tmp_lr)
    # get a batch of data
    batch_num = 1
    for iter_i, (images_, targets_) in enumerate(dataloader):
        print("Batch: %d" % batch_num)
        # new target
        targets_ = [label_.tolist() for label_ in targets_]
        targets_ = tools.gt_creator(
            input_size=train_size, stride=yolo_net.stride, label_lists=targets_
        )
        # to device
        images_ = images_.to(device)
        targets_ = torch.tensor(targets_).float().to(device)
        # forward
        conf_loss, cls_loss, bbox_loss, total_loss = model(images_, target=targets_)
        # backward
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        batch_num += 1
