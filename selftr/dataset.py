"""
author: Alessandro Nicolosi
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import glob
import os
import numpy as np
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Iterable


class ImageDataset(Dataset):
    def __init__(
        self,
        root: str,
        images: [Iterable],
        labels: Optional[Iterable] = None,
        transform: Optional[Callable] = None,
    ) -> None:

        self.root = root
        self.images = images
        self.labels = labels
        self.transform = transform

        self._checkDatasetInHardDisk(root, images)

        if self.labels is not None:
            self._checkImageLabelLen(self.images, self.labels)

    def _checkDatasetInHardDisk(self, dataset_root: str, images: Iterable) -> None:
        valid_ext = [".jpg", ".png", ".bmp", ".gif"]

        # Check that supervised images have valid extension
        for f in images:
            valid = False
            for ext in valid_ext:
                if f.endswith(ext):
                    valid = True
                    break

            if not valid:
                raise ValueError(f"{f} has an invalid extension: (valid: {valid_ext})")

        files = glob.glob(os.path.join(dataset_root, "**", "*.*"), recursive=True)

        files_rel = [os.path.relpath(f, dataset_root) for f in files]

        # Check that supervised images are physically in the hard disk
        for f in images:
            if f not in files_rel:
                raise ValueError(f"Files not found in {dataset_root}: {f}")

    def _checkImageLabelLen(self, images: Iterable, labels: Iterable) -> None:
        n_labels = len(labels)
        n_images = len(images)
        if n_labels != n_images:
            raise ValueError(
                f"labels (len: {n_labels}) must equals images (len: {n_images})"
            )

    def setLabels(self, labels: Iterable) -> None:
        self._checkImageLabelLen(self.images, labels)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        file_path = os.path.join(self.root, self.images[index])

        with open(file_path, "rb") as f:
            img = Image.open(f).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.labels is not None:
            return img, self.labels[index]

        return img
