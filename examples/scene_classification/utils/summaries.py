import os
import torch
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def creater_summary(self):
        writer = SummaryWriter(log_dir=self.directory)
        return writer

    def visualize_image(self, writer, T1, T2, label, pred, global_step, mode):
        grid_T1 = make_grid(T1[:4].clone().cpu().data, padding=50, normalize=True)
        writer.add_image(os.path.join(mode, "T1"), grid_T1, global_step)

        grid_T2 = make_grid(T2[:4].clone().cpu().data, padding=50, normalize=True)
        writer.add_image(os.path.join(mode, "T2"), grid_T2, global_step)

        grid_label = label.float()[:4, :, :, :]
        grid_label = make_grid(grid_label, padding=50, normalize=True)
        writer.add_image(os.path.join(mode, "label"), grid_label, global_step)

        grid_pred = torch.sigmoid(pred)
        grid_pred = grid_pred[:4, :, :, :]
        grid_pred = make_grid(grid_pred, padding=50, normalize=True)
        writer.add_image(os.path.join(mode, "pred"), grid_pred, global_step)
