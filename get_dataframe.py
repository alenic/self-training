import pandas as pd
import os
import numpy as np
import shutil

WRITE_FOLDER = False

root = "dataset/cifar-10/test"

classes = os.listdir(root)
all_images = []
all_labels = []

for k, c in enumerate(classes):
    path_c = os.path.join(root, c)
    images = os.listdir(path_c)

    all_images += images
    all_labels += [c for i in range(len(images))]


class_map = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9,
}


data = pd.DataFrame(columns=["image", "label"])
data.image = np.array(all_images)
data.label = np.array(all_labels)
data["label_id"] = data["label"].apply(lambda x: class_map[x])
data.to_csv("dataset/test.csv")