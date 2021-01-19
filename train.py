import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import argparse

from selftr import *


# Test dataset: to remove
import pandas as pd

supervised = pd.read_csv("dataset/supervised.csv")
unsupervised = pd.read_csv("dataset/unsupervised.csv")
# =====


def train():
    noise_transform = T.Compose(
        [
            T.RandomResizedCrop((28, 28), scale=(0.9, 1.0)),
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(0.2, 0.2, 0.2, 0.1),
            T.ToTensor(),
        ]
    )
    psudo_transform = T.Compose([T.Resize((28, 28)), T.ToTensor()])

    dataset_super = ImageDataset(
        "dataset/data",
        supervised["image"].values,
        labels=supervised["label_id"].values,
        transform=noise_transform,
    )

    dataset_pseudo = ImageDataset(
        "dataset/data",
        unsupervised["image"].values,
        transform=psudo_transform,
    )

    dataset_merged = ImageDataset(
        "dataset/data",
        list(supervised["image"].values) + list(unsupervised["image"].values),
        transform=noise_transform,
    )

    dl_super = DataLoader(
        dataset_super, batch_size=8, shuffle=True, num_workers=8, pin_memory=False
    )

    dl_pseudo = DataLoader(
        dataset_pseudo, batch_size=8, shuffle=False, num_workers=8, pin_memory=False
    )

    dl_merged = DataLoader(
        dataset_merged, batch_size=8, shuffle=True, num_workers=8, pin_memory=False
    )

    total = 0
    for image, label in dl_super:
        total += image.size()[0]
        pass

    print(total)

    total = 0
    for image in dl_pseudo:
        total += image.size()[0]
        pass

    print(total)

    total = 0
    dataset_merged.setLabels(list(supervised["label"]) + list(unsupervised["label"]))
    for image, label in dl_merged:
        total += image.size()[0]
        pass

    print(total)


if __name__ == "__main__":
    train()
