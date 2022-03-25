# ------------------------------------------------------------------------------
#
# Project: pytdml
#
# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
from torch.utils.data import Dataset
from PIL import Image


class TorchEOImageSceneTD(Dataset):
    """
    Torch Dataset for EO image scene classification training dataset
    """

    def __init__(self, td_list, class_map, transforms):
        self.td_list = td_list
        self.class_map = class_map
        self.transforms = transforms
        self.size = len(td_list)

    def __len__(self):
        return len(self.td_list)

    def __getitem__(self, item):
        img_path = self.td_list[item].data_url
        img = Image.open(img_path)
        label = self.class_map[self.td_list[item].labels[0].label_class]
        img = self.transforms(img)
        return img, label
