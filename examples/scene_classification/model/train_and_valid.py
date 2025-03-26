import time

import torch
from torch.utils.data import DataLoader


def train_and_valid(
    model,
    train_dataset,
    valid_dataset,
    train_dataset_size,
    valid_dataset_size,
    writer,
    loss_function,
    optimizer,
    epochs=1000,
    batch_size=128,
):
    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_data = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    record = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        model.train()  # train model

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(train_data):
            # inputs = inputs.cuda()
            # labels = labels.cuda()
            optimizer.zero_grad()  # zero the gradient buffers
            outputs = model(inputs)  # forward
            loss = loss_function(outputs, labels)  # loss
            loss.backward()  # 反向传播
            optimizer.step()  # 优化器更新参数
            train_loss += loss.item() * inputs.size(0)
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)

        with torch.no_grad():
            model.eval()  # 验证
            for j, (inputs, labels) in enumerate(valid_data):
                # inputs = inputs.cuda()
                # labels = labels.cuda()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                valid_acc += acc.item() * inputs.size(0)

        avg_train_loss = train_loss / train_dataset_size
        avg_train_acc = train_acc / train_dataset_size

        avg_valid_loss = valid_loss / valid_dataset_size
        avg_valid_acc = valid_acc / valid_dataset_size

        record.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        writer.add_scalar("train_loss", avg_train_loss, epoch + 1)
        writer.add_scalar("valid_loss", avg_valid_loss, epoch + 1)
        writer.add_scalar("train_acc", avg_train_acc, epoch + 1)
        writer.add_scalar("valid_acc", avg_valid_acc, epoch + 1)

        if avg_valid_acc > best_acc:  # record best accuracy and epoch
            best_acc = avg_valid_acc
            best_epoch = epoch + 1

        epoch_end = time.time()

        print(
            "Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1,
                avg_valid_loss,
                avg_train_acc * 100,
                avg_valid_loss,
                avg_valid_acc * 100,
                epoch_end - epoch_start,
            )
        )
        print(
            "Best Accuracy for validation : {:.4f} at epoch {:03d}".format(
                best_acc, best_epoch
            )
        )

        torch.save(model, "model/resnet50_model_" + str(epoch + 1) + ".pth")
    writer.close()
    return model, record
