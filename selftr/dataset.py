"""
author: Alessandro Nicolosi
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import glob
import os
import numpy as np
from typing import Any, Callable, cast, Dict, List, Optional, Tuple


class DatasetSelfTraining(Dataset):
    def __init__(
        self,
        supervised_root: str,
        supervised_images: [list, np.ndarray],
        supervised_labels: [list, np.ndarray],
        unsupervised_root: str,
        unsupervised_images: [list, np.ndarray],
        transform_phase_1: Optional[Callable] = None,
        transform_phase_2: Optional[Callable] = None,
        transform_phase_3: Optional[Callable] = None,
    ):
        self.SUPERVISED = 1
        self.PSEUDOLABEL = 2
        self.MERGED = 3

        self._state = self.SUPERVISED
        self.supervised_root = supervised_root
        self.unsupervised_root = unsupervised_root
        self.supervised_images = supervised_images
        self.supervised_labels = supervised_labels
        self.unsupervised_images = unsupervised_images
        self._pseudolabels = None
        self._merged_images = None
        self._merged_labels = None

        self.transform_phase_1 = transform_phase_1
        self.transform_phase_2 = transform_phase_2
        self.transform_phase_3 = transform_phase_3

        self._check_dataset_in_hard_disk(supervised_root, supervised_images)
        self._check_dataset_in_hard_disk(unsupervised_root, unsupervised_images)

        if len(self.supervised_labels) != len(self.supervised_images):
            raise ValueError(
                f"supervised_images (len: {len(self.supervised_images)}) not matches supervised_labels (len: {len(self.supervised_labels)}):"
            )

    def _check_dataset_in_hard_disk(self, dataset_root: str, images: list):
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

    def setPseudolabelState(self, pseudolabels: list):
        n_labels = len(pseudolabels)
        n_images = len(self.unsupervised_images)
        if n_labels != n_images:
            raise ValueError(
                f"pseudolabels (len: {n_labels}) must equals unsupervised_images (len: {n_images})"
            )
        self._pseudolabels = pseudolabels
        self._merged_images = [
            os.path.join(self.supervised_root, f) for f in self.supervised_images
        ]
        self._merged_images += [
            os.path.join(self.unsupervised_root, f) for f in self.unsupervised_images
        ]
        self._merged_labels = self.supervised_labels + self._pseudolabels

        self._state = self.PSEUDOLABEL

    def setSupervisedState(self):
        self._state = self.SUPERVISED

    def setMergedState(self):
        self._state = self.MERGED

    def __len__(self):
        return len(self.supervised_label)

    def __getitem__(self, index: int):
        if self._state == self.SUPERVISED:
            transform = self.transform_phase_1
            file_path = os.path.join(
                self.supervised_root, self.supervised_images[index]
            )
            label = self.supervised_labels[index]

        elif self._state == self.PSEUDOLABEL:
            transform = self.transform_phase_2
            file_path = os.path.join(
                self.unsupervised_root, self.unsupervised_images[index]
            )
            label = None
        elif self._state == self.MERGED:
            transform = self.transform_phase_3
            file_path = os.path.join(self._merged_images[index])
            label = self._merged_labels[index]

        with open(file_path, "rb") as f:
            img = Image.open(f)

        img = img.convert("RGB")

        if transform:
            img = transform(img)

        if label:
            return img, label

        return img
