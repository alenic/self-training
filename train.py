"""
author: Alessandro Nicolosi
website: https://github.com/alenic/self-training
"""
import argparse
import timm
import numpy as np
import pandas as pd
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from selftr import *

import sklearn.metrics as metrics

iteration = 1
date_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv-super", type=str, required=True)
    parser.add_argument("--root-super", type=str, required=True)
    parser.add_argument("--csv-val", type=str, required=True)
    parser.add_argument("--root-val", type=str, required=True)
    parser.add_argument("--csv-unsuper", type=str)
    parser.add_argument("--root-unsuper", type=str)

    parser.add_argument("--timm-model", type=str, default="resnet50")
    parser.add_argument(
        "--input-size",
        type=lambda x: [int(s) for s in x.split("x")],
        default=(229, 229),
    )
    parser.add_argument("--only-supervised", action="store_true")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-super", type=int, default=40)
    parser.add_argument("--epochs-joint", type=int, default=40)
    parser.add_argument("--drop-rate", type=float, default=0.4)
    parser.add_argument("--weight-restart", type=str, default="last")
    parser.add_argument("--cycles", type=int, default=10)

    parser.add_argument("--log-freq", type=float, default=0.25)

    args = parser.parse_args()

    assert os.path.exists(args.csv_super)
    assert os.path.isdir(args.root_super)
    assert os.path.exists(args.csv_val)
    assert os.path.isdir(args.root_val)
    if not args.only_supervised:
        assert os.path.exists(args.csv_unsuper)
        assert os.path.exists(args.root_unsuper)

    return args


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
    metric_dict = {}
    model.eval()
    pred_list = []
    label_list = []
    with torch.no_grad():
        for images, labels in data_loader:
            logits = model(images.cuda())
            pred_labels = np.argmax(logits.cpu().numpy(), axis=1)
            pred_list += list(pred_labels)
            label_list += list(labels.cpu().numpy())

        metric_dict["accuracy"] = metrics.accuracy_score(label_list, pred_list)
        metric_dict["macro_precision"] = metrics.precision_score(
            label_list, pred_list, average="macro"
        )
        metric_dict["macro_recall"] = metrics.recall_score(
            label_list, pred_list, average="macro"
        )
        metric_dict["macro_f1"] = metrics.f1_score(
            label_list, pred_list, average="macro"
        )

        print(
            f'Val acc. {metric_dict["accuracy"]*100} %, meanPrecision {metric_dict["macro_precision"]}, meanRecall {metric_dict["macro_recall"]}, meanF1 {metric_dict["macro_f1"]}'
        )

    return metric_dict


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: torch.nn.Module,
    train_data_loader: torch.utils.data.DataLoader,
    val_data_loader: torch.utils.data.DataLoader,
    phase: str,
    cycle: int = 0,
    epochs: int = 40,
    batch_size: int = 32,
    log_freq: float = 0.25,
    save_every: int = 1,
    device: str = "cuda:0",
) -> None:

    global iteration, date_now

    num_samples = len(train_data_loader.dataset)
    num_batches = num_samples // batch_size
    log_step = max(int(log_freq * num_batches), 1)

    model.train()
    model.to(device)
    for epoch in range(1, epochs + 1):
        for batch, (images, labels) in enumerate(train_data_loader):
            optimizer.zero_grad()

            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)

            loss = criterion(logits, labels)
            loss.backward()

            optimizer.step()

            iteration += 1
            if batch % log_step == 0:
                print(
                    f"{cycle} - {phase} - [{batch}/{num_batches}] Epoch {epoch} : Train loss {loss.item()}"
                )

        scheduler.step()

        metrics = eval(model, val_data_loader, device=device)

        # Save checkpoints
        if epoch % save_every == 0:
            pth_name = (
                f'{iteration:08d}_{cycle}_{phase}_{epoch}_{metrics["accuracy"]:.2f}.pth'
            )
            print(f"Saving {pth_name}")

            if not os.path.isdir(os.path.join("checkpoints", date_now)):
                os.makedirs(os.path.join("checkpoints", date_now), exist_ok=True)

            torch.save(
                model.state_dict(), os.path.join("checkpoints", date_now, pth_name)
            )


def main():
    args = parse_args()

    print(args.__dict__)

    df_super = pd.read_csv(args.csv_super)
    df_val = pd.read_csv(args.csv_val)

    if not args.only_supervised:
        df_unsuper = pd.read_csv(args.csv_unsuper)

    # TODO : multi processing
    device = torch.device("cuda:0")

    num_classes = len(df_super["label_id"].unique())
    print(f"Number of classes: {num_classes}")

    model_student = timm.create_model(
        args.timm_model,
        pretrained=True,
        num_classes=num_classes,
        drop_rate=args.drop_rate,
    )

    model_teacher = model_student

    input_size = args.input_size

    # TODO : Implement randaugment in selftr/transforms.py
    phase1_noise_transform = T.Compose(
        [
            T.RandomResizedCrop(input_size, scale=(0.9, 1.0)),
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(0.2, 0.2, 0.2, 0.1),
            T.ToTensor(),
        ]
    )

    val_transform = T.Compose([T.Resize(input_size), T.ToTensor()])

    # ======== Dataset =========
    super_images = list(df_super["image"].values)
    super_labels = list(df_super["label_id"].values)
    val_images = list(df_val["image"].values)
    val_labels = list(df_val["label_id"].values)

    dataset_super = ImageDataset(
        "dataset/data",
        super_images,
        labels=super_labels,
        transform=phase1_noise_transform,
    )

    dataset_val = ImageDataset(
        "dataset/test",
        val_images,
        labels=val_labels,
        transform=val_transform,
    )

    dl_super = DataLoader(
        dataset_super,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    dl_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    if not args.only_supervised:
        unsuper_images = list(df_unsuper["image"].values)
        phase2_psudo_transform = T.Compose([T.Resize(input_size), T.ToTensor()])

        phase3_noise_transform = T.Compose(
            [
                T.RandomRotation(degrees=(35, -35)),
                T.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(0.5),
                T.ColorJitter(0.4, 0.4, 0.4, 0.08),
                T.ToTensor(),
            ]
        )
        dataset_pseudo = ImageDataset(
            "dataset/data",
            unsuper_images,
            transform=phase2_psudo_transform,
        )

        dataset_merged = ImageDataset(
            "dataset/data",
            super_images + unsuper_images,
            transform=phase3_noise_transform,
        )

        dl_pseudo = DataLoader(
            dataset_pseudo,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

        dl_merged = DataLoader(
            dataset_merged,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

    # Phase 1 - Train Teacher model with labeled data
    optimizer = torch.optim.Adam(model_teacher.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_super, eta_min=1e-5
    )
    criterion = torch.nn.CrossEntropyLoss()

    print("Start supervised training")
    train(
        model_teacher,
        optimizer,
        scheduler,
        criterion,
        dl_super,
        dl_val,
        phase="supervised",
        cycle=0,
        epochs=args.epochs_super,
        batch_size=args.batch_size,
        device=device,
    )

    if not args.only_supervised:

        for cycle in range(1, args.cycles + 1):
            print("cycle", cycle, "Start phase 2")
            # Phase 2 - Infer psudo-labels on unlabeled data
            unsuper_pseudolabels = pseudolabeling(
                model_teacher, dl_pseudo, device=device
            )
            dataset_merged.setLabels(super_labels + unsuper_pseudolabels)

            # Phase3 - Train equal-or-larger student model with combined and noise injected
            print("cycle", cycle, "Start phase 3")

            if args.weight_restart == "pretrained":
                model_student = timm.create_model(
                    args.timm_model,
                    pretrained=True,
                    num_classes=num_classes,
                    drop_rate=args.drop_rate,
                )

            optimizer = torch.optim.Adam(model_student.parameters(), lr=args.lr)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs_joint * args.cycles, eta_min=1e-5
            )
            criterion = torch.nn.CrossEntropyLoss()

            train(
                model_student,
                optimizer,
                scheduler,
                criterion,
                dl_merged,
                dl_val,
                phase="sup+unsup",
                cycle=cycle,
                epochs=args.epochs_joint,
                batch_size=args.batch_size,
                device=device,
            )

            # Make the student a new teacher (overwrite the teacher)
            model_teacher = model_student

        torch.save(model_student.state_dict(), "final_student.pth")


if __name__ == "__main__":
    main()
