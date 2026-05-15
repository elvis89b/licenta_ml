import copy
import os
import time
import datetime
from glob import glob

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import wandb

from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader

from model.FocusNet import FocusNet
from utils import seeding, create_dir, print_and_save, epoch_time, calculate_metrics


VARIANT_SLUG = "dynamic_spectral_uncertainty_expert_routing_ema"
VARIANT_NAME = "DGFR+BandHead+DynamicSpectralUncertaintyExpertRouting+EMA"

EMA_DECAY = 0.995
EMA_CONSISTENCY_WEIGHT = 0.05
EMA_CONFIDENCE_THRESHOLD = 0.70


def infer_modality_from_path(path):
    p = str(path).upper()

    if "BLI" in p:
        return "BLI"
    if "FICE" in p:
        return "FICE"
    if "LCI" in p:
        return "LCI"
    if "NBI" in p:
        return "NBI"
    if "WLI" in p:
        return "WLI"

    return "UNKNOWN"


def infer_center_from_path(path):
    p = str(path).replace("\\", "/")
    parts = p.split("/")

    if "PolypDB_center_wise" in parts:
        idx = parts.index("PolypDB_center_wise")

        if idx + 1 < len(parts):
            return parts[idx + 1]

    return "UNKNOWN_CENTER"


def load_polypdb_center_data(path):
    samples = []

    images_jpg = sorted(glob(os.path.join(path, "images", "*.jpg")))
    images_png = sorted(glob(os.path.join(path, "images", "*.png")))

    images = images_jpg + images_png

    for image_path in images:
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        mask_jpg = os.path.join(path, "masks", f"{image_name}.jpg")
        mask_png = os.path.join(path, "masks", f"{image_name}.png")

        if os.path.exists(mask_png):
            mask_path = mask_png
        elif os.path.exists(mask_jpg):
            mask_path = mask_jpg
        else:
            continue

        samples.append((image_path, mask_path))

    center_len = len(samples)
    center_train_len = int(0.8 * center_len)
    center_val_len = int(0.1 * center_len)

    train_samples = samples[:center_train_len]
    valid_samples = samples[
        center_train_len:center_train_len + center_val_len
    ]
    test_samples = samples[
        center_train_len + center_val_len:
    ]

    return train_samples, valid_samples, test_samples


class PolypDB_DATASET(Dataset):
    def __init__(self, samples_path, size, transform=None):
        super().__init__()

        self.samples_path = samples_path
        self.transform = transform
        self.n_samples = len(samples_path)
        self.size = size

    def __getitem__(self, index):
        image_path = self.samples_path[index][0]
        mask_path = self.samples_path[index][1]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        if mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        image = cv2.resize(
            image,
            self.size,
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            self.size,
            interpolation=cv2.INTER_NEAREST
        )

        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32) / 255.0

        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)

        modality = infer_modality_from_path(image_path)

        return image, mask, modality

    def __len__(self):
        return self.n_samples


def confidence_filtered_ema_consistency_loss(
    student_logits,
    teacher_logits,
    confidence_threshold
):
    student_probs = torch.sigmoid(student_logits)
    teacher_probs = torch.sigmoid(teacher_logits.detach())

    teacher_confidence = torch.maximum(
        teacher_probs,
        1.0 - teacher_probs
    )

    confidence_mask = (
        teacher_confidence >= confidence_threshold
    ).float()

    consistency_map = F.mse_loss(
        student_probs,
        teacher_probs,
        reduction="none"
    )

    consistency_loss = (
        consistency_map * confidence_mask
    ).sum() / (
        confidence_mask.sum() + 1e-6
    )

    return consistency_loss


def update_ema_model(student_model, ema_model, decay):
    with torch.no_grad():
        for ema_param, student_param in zip(
            ema_model.parameters(),
            student_model.parameters()
        ):
            ema_param.data.mul_(decay).add_(
                student_param.data,
                alpha=1.0 - decay
            )

        for ema_buffer, student_buffer in zip(
            ema_model.buffers(),
            student_model.buffers()
        ):
            ema_buffer.copy_(student_buffer)


def train(model, ema_model, loader, optimizer, device):
    model.train()
    ema_model.eval()

    epoch_loss = 0.0
    epoch_model_loss = 0.0
    epoch_ema_consistency = 0.0

    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    loss_accumulator = {
        "loss_final": 0.0,
        "loss_band": 0.0,
        "loss_dynamic_expert": 0.0,
        "loss_ugel": 0.0,
        "loss_spectral": 0.0,
        "loss_region": 0.0,
        "loss_consistency": 0.0,
        "loss_router_entropy": 0.0,
        "router_ugel_weight": 0.0,
        "router_spectral_weight": 0.0,
        "router_region_weight": 0.0
    }

    for x, y, modalities in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        sample = {
            "images": x,
            "masks": y,
            "modalities": modalities
        }

        student_out = model(sample)

        with torch.no_grad():
            teacher_out = ema_model(sample)

        ema_consistency_loss = confidence_filtered_ema_consistency_loss(
            student_logits=student_out["prediction"],
            teacher_logits=teacher_out["prediction"],
            confidence_threshold=EMA_CONFIDENCE_THRESHOLD
        )

        model_loss = student_out["loss"]

        total_loss = (
            model_loss
            + EMA_CONSISTENCY_WEIGHT * ema_consistency_loss
        )

        total_loss.backward()
        optimizer.step()

        update_ema_model(
            student_model=model,
            ema_model=ema_model,
            decay=EMA_DECAY
        )

        epoch_loss += total_loss.item()
        epoch_model_loss += model_loss.item()
        epoch_ema_consistency += ema_consistency_loss.item()

        for key in loss_accumulator.keys():
            loss_accumulator[key] += student_out[key].item()

        y_pred = torch.sigmoid(student_out["prediction"])

        batch_jac = []
        batch_f1 = []
        batch_recall = []
        batch_precision = []

        for yt, yp in zip(y, y_pred):
            score = calculate_metrics(yt, yp)

            batch_jac.append(score[0])
            batch_f1.append(score[1])
            batch_recall.append(score[2])
            batch_precision.append(score[3])

        epoch_jac += np.mean(batch_jac)
        epoch_f1 += np.mean(batch_f1)
        epoch_recall += np.mean(batch_recall)
        epoch_precision += np.mean(batch_precision)

    n_batches = len(loader)

    metrics = [
        epoch_jac / n_batches,
        epoch_f1 / n_batches,
        epoch_recall / n_batches,
        epoch_precision / n_batches
    ]

    loss_parts = {
        "total_loss": epoch_loss / n_batches,
        "model_loss": epoch_model_loss / n_batches,
        "ema_consistency": epoch_ema_consistency / n_batches
    }

    for key in loss_accumulator.keys():
        loss_parts[key] = loss_accumulator[key] / n_batches

    return epoch_loss / n_batches, metrics, loss_parts


def evaluate(model, loader, device):
    model.eval()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    loss_accumulator = {
        "loss_final": 0.0,
        "loss_band": 0.0,
        "loss_dynamic_expert": 0.0,
        "loss_ugel": 0.0,
        "loss_spectral": 0.0,
        "loss_region": 0.0,
        "loss_consistency": 0.0,
        "loss_router_entropy": 0.0,
        "router_ugel_weight": 0.0,
        "router_spectral_weight": 0.0,
        "router_region_weight": 0.0
    }

    with torch.no_grad():
        for x, y, modalities in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            sample = {
                "images": x,
                "masks": y,
                "modalities": modalities
            }

            out = model(sample)

            epoch_loss += out["loss"].item()

            for key in loss_accumulator.keys():
                loss_accumulator[key] += out[key].item()

            y_pred = torch.sigmoid(out["prediction"])

            batch_jac = []
            batch_f1 = []
            batch_recall = []
            batch_precision = []

            for yt, yp in zip(y, y_pred):
                score = calculate_metrics(yt, yp)

                batch_jac.append(score[0])
                batch_f1.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])

            epoch_jac += np.mean(batch_jac)
            epoch_f1 += np.mean(batch_f1)
            epoch_recall += np.mean(batch_recall)
            epoch_precision += np.mean(batch_precision)

    n_batches = len(loader)

    metrics = [
        epoch_jac / n_batches,
        epoch_f1 / n_batches,
        epoch_recall / n_batches,
        epoch_precision / n_batches
    ]

    loss_parts = {}

    for key in loss_accumulator.keys():
        loss_parts[key] = loss_accumulator[key] / n_batches

    return epoch_loss / n_batches, metrics, loss_parts


def run_experiment(path):
    model_name = "FocusNet"

    current_center = infer_center_from_path(path)
    current_modality = infer_modality_from_path(path)

    experiment_name = (
        f"FocusNet_DGFR_BandHead_"
        f"DynamicSpectralUncertaintyExpertRouting_EMA_"
        f"center_{current_center}_{current_modality}"
    )

    create_dir("files")
    create_dir(f"files/center_wise/{model_name}")

    train_log_path = (
        f"files/center_wise/{model_name}/"
        f"train_log_{current_center}_{current_modality}_{VARIANT_SLUG}.txt"
    )

    with open(train_log_path, "w") as train_log:
        train_log.write("")

    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)
    print("")

    image_size = 256
    size = (image_size, image_size)

    batch_size = 16
    num_epochs = 500
    lr = 1e-4
    weight_decay = 1e-4
    early_stopping_patience = 50

    checkpoint_path = (
        f"files/center_wise/{model_name}/"
        f"checkpoint_{current_center}_{current_modality}_{VARIANT_SLUG}.pth"
    )

    wandb.init(
        project="polyp-segmentation-focusnet",
        name=experiment_name,
        reinit=True,
        settings=wandb.Settings(init_timeout=180),
        config={
            "model": model_name,
            "variant": VARIANT_NAME,
            "setting": "center_wise",
            "center": current_center,
            "modality": current_modality,
            "image_size": image_size,
            "batch_size": batch_size,
            "epochs": num_epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "early_stopping_patience": early_stopping_patience,
            "ema_decay": EMA_DECAY,
            "ema_consistency_weight": EMA_CONSISTENCY_WEIGHT,
            "ema_confidence_threshold": EMA_CONFIDENCE_THRESHOLD,
            "loss": "Dynamic UGEL + Spectral Boundary + Region Calibration Router + EMA",
            "augmentation": "Spatial Aug only",
            "checkpoint_model": "EMA teacher"
        }
    )

    data_str = f"Experiment: {experiment_name}\n"
    data_str += f"Variant: {VARIANT_NAME}\n"
    data_str += f"Setting: center_wise\n"
    data_str += f"Center: {current_center}\n"
    data_str += f"Modality: {current_modality}\n"
    data_str += f"Image Size: {size}\n"
    data_str += f"Batch Size: {batch_size}\n"
    data_str += f"LR: {lr}\n"
    data_str += f"Weight Decay: {weight_decay}\n"
    data_str += f"Epochs: {num_epochs}\n"
    data_str += f"Early Stopping Patience: {early_stopping_patience}\n"
    data_str += f"EMA Decay: {EMA_DECAY}\n"
    data_str += f"EMA Consistency Weight: {EMA_CONSISTENCY_WEIGHT}\n"
    data_str += f"EMA Confidence Threshold: {EMA_CONFIDENCE_THRESHOLD}\n"
    data_str += f"Path: {path}\n"
    data_str += f"Checkpoint: {checkpoint_path}\n"

    print_and_save(train_log_path, data_str)

    train_samples, valid_samples, test_samples = load_polypdb_center_data(path)
    train_samples = shuffle(train_samples, random_state=42)

    data_str = f"Dataset Size:\n"
    data_str += (
        f"Train: {len(train_samples)} - "
        f"Valid: {len(valid_samples)} - "
        f"Test: {len(test_samples)}\n"
    )

    print_and_save(train_log_path, data_str)

    if len(train_samples) == 0 or len(valid_samples) == 0:
        print_and_save(
            train_log_path,
            f"Skipping {current_center}/{current_modality}: empty train or validation split.\n"
        )
        wandb.finish()
        return

    transform = A.Compose([
        A.Rotate(limit=35, p=0.30),
        A.HorizontalFlip(p=0.30),
        A.VerticalFlip(p=0.30),
        A.CoarseDropout(
            max_holes=8,
            max_height=24,
            max_width=24,
            p=0.20
        )
    ])

    train_dataset = PolypDB_DATASET(
        samples_path=train_samples,
        size=size,
        transform=transform
    )

    valid_dataset = PolypDB_DATASET(
        samples_path=valid_samples,
        size=size,
        transform=None
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FocusNet().to(device)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()

    for param in ema_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=5,
        verbose=True
    )

    best_valid_f1 = 0.0
    early_stopping_count = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_metrics, train_parts = train(
            model=model,
            ema_model=ema_model,
            loader=train_loader,
            optimizer=optimizer,
            device=device
        )

        valid_loss, valid_metrics, valid_parts = evaluate(
            model=ema_model,
            loader=valid_loader,
            device=device
        )

        scheduler.step(valid_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        if valid_metrics[1] > best_valid_f1:
            data_str = (
                f"Valid F1 improved from {best_valid_f1:2.4f} "
                f"to {valid_metrics[1]:2.4f}. "
                f"Saving EMA checkpoint: {checkpoint_path}"
            )

            print_and_save(train_log_path, data_str)

            best_valid_f1 = valid_metrics[1]
            torch.save(ema_model.state_dict(), checkpoint_path)
            early_stopping_count = 0
        else:
            early_stopping_count += 1

        wandb.log({
            "epoch": epoch + 1,
            "lr": current_lr,

            "train/loss": train_loss,
            "train/jaccard": train_metrics[0],
            "train/f1": train_metrics[1],
            "train/recall": train_metrics[2],
            "train/precision": train_metrics[3],

            "valid/loss": valid_loss,
            "valid/jaccard": valid_metrics[0],
            "valid/f1": valid_metrics[1],
            "valid/recall": valid_metrics[2],
            "valid/precision": valid_metrics[3],

            "train/model_loss": train_parts["model_loss"],
            "train/ema_consistency": train_parts["ema_consistency"],
            "train/router_ugel_weight": train_parts["router_ugel_weight"],
            "train/router_spectral_weight": train_parts["router_spectral_weight"],
            "train/router_region_weight": train_parts["router_region_weight"],

            "valid/router_ugel_weight": valid_parts["router_ugel_weight"],
            "valid/router_spectral_weight": valid_parts["router_spectral_weight"],
            "valid/router_region_weight": valid_parts["router_region_weight"],

            "best_valid_f1": best_valid_f1
        })

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"

        data_str += (
            f"\tTrain Loss: {train_loss:.4f} "
            f"- Jaccard: {train_metrics[0]:.4f} "
            f"- F1: {train_metrics[1]:.4f} "
            f"- Recall: {train_metrics[2]:.4f} "
            f"- Precision: {train_metrics[3]:.4f}\n"
        )

        data_str += (
            f"\t Val. Loss: {valid_loss:.4f} "
            f"- Jaccard: {valid_metrics[0]:.4f} "
            f"- F1: {valid_metrics[1]:.4f} "
            f"- Recall: {valid_metrics[2]:.4f} "
            f"- Precision: {valid_metrics[3]:.4f}\n"
        )

        print_and_save(train_log_path, data_str)

        if early_stopping_count == early_stopping_patience:
            data_str = (
                f"Early stopping: validation F1 stopped improving for "
                f"{early_stopping_patience} consecutive epochs.\n"
            )

            print_and_save(train_log_path, data_str)
            break

    wandb.finish()

    del model
    del ema_model
    del optimizer
    del scheduler

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    seeding(42)

    center_paths = [
        "data/PolypDB/PolypDB_center_wise/Simula/WLI",
        "data/PolypDB/PolypDB_center_wise/BKAI/WLI",
        "data/PolypDB/PolypDB_center_wise/Karolinska/WLI",
    ]

    for path in center_paths:
        if not os.path.exists(path):
            print(f"Skipping missing path: {path}")
            continue

        print("\n" + "=" * 100)
        print(f"Starting center-wise experiment for: {path}")
        print("=" * 100 + "\n")

        run_experiment(path)
