import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self, images_path: list, images_label: list, transform=None):
        self.images_path = images_path
        self.images_label = images_label
        self.transform = transform

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])

        label = self.images_label[item]

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images_path)

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
