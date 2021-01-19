import torch
import argparse

from selftr import *


# Test dataset: to remove
import pandas as pd

supervised = pd.read_csv("dataset/supervised.csv")
unsupervised = pd.read_csv("dataset/unsupervised.csv")
# =====


def train():
    # test
    dataset = DatasetSelfTraining(
        "dataset/supervised",
        supervised["image"].values,
        supervised["label_id"].values,
        "dataset/unsupervised",
        unsupervised["image"].values,
    )

    print(dataset.supervised_images[:10], dataset.supervised_labels[:10])


if __name__ == "__main__":
    train()
