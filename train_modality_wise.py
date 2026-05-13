import copy
import os
import time
import datetime
from glob import glob

import albumentations as A
import cv2
import numpy as np
import torch
import wandb

from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader

from model.FocusNet import FocusNet
from utils import seeding, create_dir, print_and_save, epoch_time, calculate_metrics


VARIANT_SLUG = "region_adaptive_hybrid_supervision_ema"
VARIANT_NAME = "DGFR+BandHead+RegionAdaptiveHybridSupervision+EMA"
EMA_DECAY = 0.995


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


def load_polypdb_modality_data(path):
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

    modality_len = len(samples)
    modality_train_len = int(0.8 * modality_len)
    modality_val_len = int(0.1 * modality_len)

    train_samples = samples[:modality_train_len]
    valid_samples = samples[modality_train_len:modality_train_len + modality_val_len]
    test_samples = samples[modality_train_len + modality_val_len:]

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

        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)

        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32) / 255.0

        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)

        modality = infer_modality_from_path(image_path)
        center = "UNKNOWN_CENTER"

        return image, mask, modality, center

    def __len__(self):
        return self.n_samples


def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def update_ema_model(student_model, ema_model, decay):
    with torch.no_grad():
        student_state = student_model.state_dict()
        ema_state = ema_model.state_dict()

        for key in ema_state.keys():
            if not torch.is_floating_point(ema_state[key]):
                ema_state[key].copy_(student_state[key])
            else:
                ema_state[key].mul_(decay).add_(student_state[key], alpha=1.0 - decay)


def train(model, ema_model, loader, optimizer, device):
    model.train()
    ema_model.eval()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    loss_sums = {
        "loss_final": 0.0,
        "loss_band": 0.0,
        "loss_ugel": 0.0,
        "loss_wavelet": 0.0,
        "loss_safe_background": 0.0,
        "loss_focal_tversky": 0.0,
        "loss_consistency": 0.0,
        "loss_teacher": 0.0,
    }

    for x, y, modalities, centers in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        with torch.no_grad():
            teacher_sample = {
                "images": x,
                "masks": y,
                "modalities": modalities,
                "centers": centers,
            }
            teacher_out = ema_model(teacher_sample)
            teacher_prediction = teacher_out["prediction"].detach()

        optimizer.zero_grad()

        sample = {
            "images": x,
            "masks": y,
            "modalities": modalities,
            "centers": centers,
            "teacher_prediction": teacher_prediction,
        }

        out = model(sample)
        y_pred = out["prediction"]
        loss = out["loss"]

        loss.backward()
        optimizer.step()
        update_ema_model(model, ema_model, EMA_DECAY)

        epoch_loss += loss.item()
        for key in loss_sums.keys():
            loss_sums[key] += out[key].item()

        batch_jac = []
        batch_f1 = []
        batch_recall = []
        batch_precision = []

        y_pred = torch.sigmoid(y_pred)

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
    epoch_loss /= n_batches
    epoch_jac /= n_batches
    epoch_f1 /= n_batches
    epoch_recall /= n_batches
    epoch_precision /= n_batches

    loss_parts = {key: value / n_batches for key, value in loss_sums.items()}

    return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision], loss_parts


def evaluate(model, loader, device):
    model.eval()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    loss_sums = {
        "loss_final": 0.0,
        "loss_band": 0.0,
        "loss_ugel": 0.0,
        "loss_wavelet": 0.0,
        "loss_safe_background": 0.0,
        "loss_focal_tversky": 0.0,
        "loss_consistency": 0.0,
        "loss_teacher": 0.0,
    }

    with torch.no_grad():
        for x, y, modalities, centers in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            sample = {
                "images": x,
                "masks": y,
                "modalities": modalities,
                "centers": centers,
            }

            out = model(sample)
            y_pred = out["prediction"]
            loss = out["loss"]

            epoch_loss += loss.item()
            for key in loss_sums.keys():
                loss_sums[key] += out[key].item()

            batch_jac = []
            batch_f1 = []
            batch_recall = []
            batch_precision = []

            y_pred = torch.sigmoid(y_pred)

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
    epoch_loss /= n_batches
    epoch_jac /= n_batches
    epoch_f1 /= n_batches
    epoch_recall /= n_batches
    epoch_precision /= n_batches

    loss_parts = {key: value / n_batches for key, value in loss_sums.items()}

    return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision], loss_parts


def run_experiment(path):
    model_name = "FocusNet"
    current_modality = infer_modality_from_path(path)

    experiment_name = (
        f"FocusNet_DGFR_BandHead_RegionAdaptiveHybridSupervision_EMA_"
        f"modality_{current_modality}"
    )

    create_dir("files")
    create_dir(f"files/modality_wise/{model_name}")

    train_log_path = (
        f"files/modality_wise/{model_name}/"
        f"train_log_{current_modality}_{VARIANT_SLUG}.txt"
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
        f"files/modality_wise/{model_name}/"
        f"checkpoint_{current_modality}_{VARIANT_SLUG}.pth"
    )

    wandb.init(
        project="polyp-segmentation-focusnet",
        name=experiment_name,
        reinit=True,
        settings=wandb.Settings(init_timeout=180),
        config={
            "model": model_name,
            "variant": VARIANT_NAME,
            "setting": "modality_wise",
            "modality": current_modality,
            "image_size": image_size,
            "batch_size": batch_size,
            "epochs": num_epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "early_stopping_patience": early_stopping_patience,
            "ema_decay": EMA_DECAY,
            "train_path": path,
            "loss": "Region-adaptive DiceBCE + Band + UGEL + GrayWavelet + SafeBG + FT + EMA",
            "augmentation": "Spatial Aug only",
        }
    )

    data_str = f"Experiment: {experiment_name}\n"
    data_str += f"Variant: {VARIANT_NAME}\n"
    data_str += f"Setting: modality_wise\n"
    data_str += f"Modality: {current_modality}\n"
    data_str += f"Image Size: {size}\n"
    data_str += f"Batch Size: {batch_size}\n"
    data_str += f"LR: {lr}\n"
    data_str += f"Weight Decay: {weight_decay}\n"
    data_str += f"Epochs: {num_epochs}\n"
    data_str += f"Early Stopping Patience: {early_stopping_patience}\n"
    data_str += f"EMA Decay: {EMA_DECAY}\n"
    data_str += f"Path: {path}\n"
    data_str += f"Checkpoint: {checkpoint_path}\n"
    print_and_save(train_log_path, data_str)

    train_samples, valid_samples, _ = load_polypdb_modality_data(path)
    train_samples = shuffle(train_samples, random_state=42)

    data_str = f"Dataset Size:\n"
    data_str += f"Train: {len(train_samples)} - Valid: {len(valid_samples)}\n"
    print_and_save(train_log_path, data_str)

    if len(train_samples) == 0 or len(valid_samples) == 0:
        print_and_save(train_log_path, f"Skipping {current_modality}: empty train or validation split.\n")
        wandb.finish()
        return

    transform = A.Compose([
        A.Rotate(limit=35, p=0.30),
        A.HorizontalFlip(p=0.30),
        A.VerticalFlip(p=0.30),
        A.CoarseDropout(max_holes=8, max_height=24, max_width=24, p=0.20),
    ])

    train_dataset = PolypDB_DATASET(train_samples, size=size, transform=transform)
    valid_dataset = PolypDB_DATASET(valid_samples, size=size, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FocusNet().to(device)
    ema_model = copy.deepcopy(model).to(device)
    set_requires_grad(ema_model, False)
    ema_model.eval()

    print(f"train model: {model_name}")
    print(f"experiment: {experiment_name}")
    print(f"path: {path}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, verbose=True)

    data_str = "Optimizer: AdamW\n"
    data_str += "Scheduler: ReduceLROnPlateau\n"
    data_str += "Loss: Region-adaptive DiceBCE + Band + UGEL + GrayWavelet + SafeBG + FT + EMA\n"
    data_str += "Augmentation: Spatial Aug only\n"
    print_and_save(train_log_path, data_str)

    best_valid_f1 = 0.0
    early_stopping_count = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_metrics, train_loss_parts = train(model, ema_model, train_loader, optimizer, device)
        valid_loss, valid_metrics, valid_loss_parts = evaluate(ema_model, valid_loader, device)

        scheduler.step(valid_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if valid_metrics[1] > best_valid_f1:
            data_str = (
                f"Valid EMA F1 improved from {best_valid_f1:2.4f} "
                f"to {valid_metrics[1]:2.4f}. Saving checkpoint: {checkpoint_path}"
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
            "best_valid_f1": best_valid_f1,
            **{f"train/{k}": v for k, v in train_loss_parts.items()},
            **{f"valid/{k}": v for k, v in valid_loss_parts.items()},
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
        data_str += (
            f"\t Valid Loss Parts "
            f"- Final: {valid_loss_parts['loss_final']:.4f} "
            f"- Band: {valid_loss_parts['loss_band']:.4f} "
            f"- UGEL: {valid_loss_parts['loss_ugel']:.4f} "
            f"- Wavelet: {valid_loss_parts['loss_wavelet']:.4f} "
            f"- SafeBG: {valid_loss_parts['loss_safe_background']:.4f} "
            f"- FT: {valid_loss_parts['loss_focal_tversky']:.4f} "
            f"- Consistency: {valid_loss_parts['loss_consistency']:.4f} "
            f"- Teacher: {valid_loss_parts['loss_teacher']:.4f}\n"
        )
        print_and_save(train_log_path, data_str)

        if early_stopping_count == early_stopping_patience:
            data_str = f"Early stopping: validation F1 stopped improving for {early_stopping_patience} consecutive epochs.\n"
            print_and_save(train_log_path, data_str)
            break

    wandb.finish()
    del model, ema_model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    seeding(42)

    modality_paths = [
        "data/PolypDB/PolypDB_modality_wise/WLI",
        "data/PolypDB/PolypDB_modality_wise/BLI",
        "data/PolypDB/PolypDB_modality_wise/FICE",
        "data/PolypDB/PolypDB_modality_wise/LCI",
        "data/PolypDB/PolypDB_modality_wise/NBI",
    ]

    for path in modality_paths:
        if not os.path.exists(path):
            print(f"Skipping missing path: {path}")
            continue

        print("\n" + "=" * 100)
        print(f"Starting modality-wise experiment for: {path}")
        print("=" * 100 + "\n")
        run_experiment(path)
