from torchvision import transforms

import pytdml
import torch
import datetime
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

from modelNet.deeplabv3 import DeepLabV3
from modelNet.deeplab_util import add_weight_decay, label_accuracy_score
from pytdml.utils import class_to_index

# Load the training dataset
training_dataset = pytdml.io.read_from_json("gid_5c.json")  # read from TDML json file
print("Load training dataset: " + training_dataset.name)
print("Number of training samples: " + str(training_dataset.amount_of_training_data))
print("Number of classes: " + str(training_dataset.number_of_classes))

# Prepare the training dataset
class_map = pytdml.ml.creat_class_map(training_dataset)  # create class map
test = class_to_index(class_map)
train_set, val_set, test_set = pytdml.ml.split_train_valid_test(training_dataset, 0.7, 0.2, 0.1)  # split dataset
train_dataset = pytdml.ml.TorchEOImageSegmentationTD(  # create Torch train dataset
    train_set,
    class_map,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
)
val_dataset = pytdml.ml.TorchEOImageSegmentationTD(
    val_set,
    class_map,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
)
batch_size = 16
train_data = data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)
val_data = data.DataLoader(val_dataset, 128, num_workers=0)

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
    for iter_i, data in enumerate(train_data):
        im = data[0]
        label = data[1]
        # forward
        out = net(im)
        out = F.log_softmax(out, dim=1)  # (b, n, h, w)
        loss = criterion(out, label)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        label_pred = out.max(dim=1)[1].data.cpu().numpy()
        label_true = label.data.cpu().numpy()
        for lbt, lbp in zip(label_true, label_pred):
            acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, 6)
            train_acc += acc
            train_acc_cls += acc_cls
            train_mean_iu += mean_iu
            train_fwavacc += fwavacc
        net = net.eval()
        eval_loss = 0
        eval_acc = 0
        eval_acc_cls = 0
        eval_mean_iu = 0
        eval_fwavacc = 0
        with torch.no_grad():
            for data in val_data:
                im = data[0]
                label = data[1]
                # forward
                out = net(im)
                out = F.log_softmax(out, dim=1)
                loss = criterion(out, label)
                eval_loss += loss.item()
                label_pred = out.max(dim=1)[1].data.cpu().numpy()
                label_true = label.data.cpu().numpy()
                for lbt, lbp in zip(label_true, label_pred):
                    acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, 6)
                    eval_acc += acc
                    eval_acc_cls += acc_cls
                    eval_mean_iu += mean_iu
                    eval_fwavacc += fwavacc
            epoch_size = len(train_dataset) // batch_size
        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        epoch_str = (
            'Epoch: {}, Train Loss: {:.5f}, Train Acc: {:.5f}, Train Mean IU: {:.5f},   Val Loss: {:.5f}, Val Acc: {:.5f}, Val Mean IU: {:.5f} '.format(
                e + 1, train_loss / len(train_data), train_acc / len(train_dataset), train_mean_iu / len(train_dataset),
                eval_loss / len(val_data), eval_acc / len(val_dataset), eval_mean_iu / len(val_dataset)))
        time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
        print(epoch_str + time_str)  # + ' lr: {}'.format(optimizer.learning_rate)
        print('------------------------------------------------------------------------------------------------------')
