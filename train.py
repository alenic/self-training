import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

import argparse
import timm
import numpy as np

from selftr import *

import sklearn.metrics as metrics


# Test dataset: to remove
import pandas as pd
import time

supervised = pd.read_csv("dataset/supervised.csv")
unsupervised = pd.read_csv("dataset/unsupervised.csv")
test = pd.read_csv("dataset/test.csv")

device = torch.device("cuda:0")
# =====

__LR = 0.001
__BATCH_SIZE = 64
__PH1_EPOCHS = 30
__PH3_EPOCHS = 20
__CYCLES = 5
__LOG_FREQ = 0.25

iteration = 1


def pseudolabeling(
    model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device="cuda:0"
) -> list:
    unsuper_pseudolabels = []
    model.eval()
    model.to(device)
    with torch.no_grad():
        for image in data_loader:
            image = image.to(device)
            logits = model(image).cpu().numpy()
            pseudolabels = np.argmax(logits, 1)
            unsuper_pseudolabels += list(pseudolabels)

    return unsuper_pseudolabels


def eval(
    model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device="cuda:0"
):
    model.eval()
    pred_list = []
    label_list = []
    with torch.no_grad():
        for images, labels in data_loader:
            logits = model(images.cuda())
            pred_labels = np.argmax(logits.cpu().numpy(), axis=1)
            pred_list += list(pred_labels)
            label_list += list(labels.cpu().numpy())

        val_acc = metrics.accuracy_score(label_list, pred_list)
        macro_precision = metrics.precision_score(
            label_list, pred_list, average="macro"
        )
        macro_recall = metrics.recall_score(label_list, pred_list, average="macro")
        macro_f1 = metrics.f1_score(label_list, pred_list, average="macro")

        print(
            f"Val acc. {val_acc*100} %, meanPrecision {macro_precision}, meanRecall {macro_recall}, meanF1 {macro_f1}"
        )


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: torch.nn.Module,
    train_data_loader: torch.utils.data.DataLoader,
    val_data_loader: torch.utils.data.DataLoader,
    phase: int,
    epochs: int,
    batch_size: int,
    device="cuda:0",
) -> None:
    global iteration
    num_samples = len(train_data_loader.dataset)
    log_step = max(int(__LOG_FREQ * num_samples / batch_size), 1)

    model.train()
    model.to(device)
    for epoch in range(1, epochs + 1):
        for images, labels in train_data_loader:
            optimizer.zero_grad()

            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)

            loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()

            iteration += 1
            if iteration % log_step == 0:
                print(f"Phase {phase} Epoch {epoch} : Train loss {loss.item()}")

        scheduler.step()
        # if epoch % 5 == 0:
        #    torch.save(model.state_dict(), "model_%d.pth" % (epoch + 1))
        eval(model, val_data_loader, device=device)


def start():
    # model_student = timm.create_model(
    #    "resnet18", pretrained=True, num_classes=10, drop_rate=0.4
    # )
    model_student = CNNSuper(num_classes=10)
    model_teacher = model_student

    input_size = (28, 28)

    noise_transform = T.Compose(
        [
            T.RandomResizedCrop(input_size, scale=(0.9, 1.0)),
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(0.2, 0.2, 0.2, 0.1),
            T.ToTensor(),
        ]
    )
    psudo_transform = T.Compose([T.Resize(input_size), T.ToTensor()])

    # ======== Dataset =========
    super_images = list(supervised["image"].values)
    super_labels = list(supervised["label_id"].values)
    unsuper_images = list(unsupervised["image"].values)
    val_images = list(test["image"].values)
    val_labels = list(test["label_id"].values)

    dataset_super = ImageDataset(
        "dataset/data",
        super_images,
        labels=super_labels,
        transform=noise_transform,
    )

    dataset_pseudo = ImageDataset(
        "dataset/data",
        unsuper_images,
        transform=psudo_transform,
    )

    dataset_merged = ImageDataset(
        "dataset/data",
        super_images + unsuper_images,
        transform=noise_transform,
    )

    dataset_val = ImageDataset(
        "dataset/test",
        val_images,
        labels=val_labels,
        transform=psudo_transform,
    )

    dl_super = DataLoader(
        dataset_super,
        batch_size=__BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    dl_pseudo = DataLoader(
        dataset_pseudo,
        batch_size=__BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    dl_merged = DataLoader(
        dataset_merged,
        batch_size=__BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    dl_val = DataLoader(
        dataset_val,
        batch_size=__BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # Phase 1 - Train Teacher model with labeled data
    optimizer = torch.optim.Adam(model_teacher.parameters(), lr=__LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=__PH1_EPOCHS, eta_min=1e-5
    )
    criterion = torch.nn.CrossEntropyLoss()

    print("Start phase 1")
    train(
        model_teacher,
        optimizer,
        scheduler,
        criterion,
        dl_super,
        dl_val,
        phase=1,
        epochs=__PH1_EPOCHS,
        batch_size=__BATCH_SIZE,
        device=device,
    )

    for i in range(1, __CYCLES + 1):
        print("cycle", i, "Start phase 2")
        # Phase 2 - Infer psudo-labels on unlabeled data
        unsuper_pseudolabels = pseudolabeling(model_teacher, dl_pseudo, device=device)
        dataset_merged.setLabels(super_labels + unsuper_pseudolabels)

        # Phase3 - Train equal-or-larger student model with combined and noise injected
        print("cycle", i, "Start phase 3")
        optimizer = torch.optim.Adam(model_student.parameters(), lr=__LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=__PH3_EPOCHS, eta_min=1e-5
        )
        criterion = torch.nn.CrossEntropyLoss()

        train(
            model_student,
            optimizer,
            scheduler,
            criterion,
            dl_merged,
            dl_val,
            phase=3,
            epochs=__PH3_EPOCHS,
            batch_size=__BATCH_SIZE,
            device=device,
        )

        # Make the student a new teacher (overwrite the teacher)
        model_teacher = model_student

    torch.save(model_student.state_dict(), "final_student.pth")


if __name__ == "__main__":
    start()
