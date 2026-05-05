import os
import time
import datetime
import random
from glob import glob

import cv2
import numpy as np
import albumentations as A

import torch
from torch.utils.data import Dataset, DataLoader

import wandb
from sklearn.utils import shuffle

from model.FocusNet import FocusNet
from utils import seeding, create_dir, print_and_save, epoch_time, calculate_metrics


EXPERIMENT_TAG = "ugel_freqampmix"
VARIANT_NAME = "DGFR+BandHead+UGEL+FreqAmpMix"


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

        mask_png = os.path.join(path, "masks", f"{image_name}.png")
        mask_jpg = os.path.join(path, "masks", f"{image_name}.jpg")

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


def freq_amp_mix(image, reference_image, alpha=0.25, low_freq_ratio=0.10):
    image = image.astype(np.float32) / 255.0
    reference_image = reference_image.astype(np.float32) / 255.0

    mixed = np.zeros_like(image, dtype=np.float32)

    h, w, c = image.shape
    radius = max(1, int(min(h, w) * low_freq_ratio))
    center_h, center_w = h // 2, w // 2

    y1 = max(0, center_h - radius)
    y2 = min(h, center_h + radius)
    x1 = max(0, center_w - radius)
    x2 = min(w, center_w + radius)

    for ch in range(c):
        src_fft = np.fft.fft2(image[:, :, ch])
        ref_fft = np.fft.fft2(reference_image[:, :, ch])

        src_amp = np.abs(src_fft)
        src_phase = np.angle(src_fft)

        ref_amp = np.abs(ref_fft)

        src_amp_shift = np.fft.fftshift(src_amp)
        ref_amp_shift = np.fft.fftshift(ref_amp)

        src_amp_shift[y1:y2, x1:x2] = (
            (1.0 - alpha) * src_amp_shift[y1:y2, x1:x2]
            + alpha * ref_amp_shift[y1:y2, x1:x2]
        )

        src_amp_mixed = np.fft.ifftshift(src_amp_shift)
        mixed_fft = src_amp_mixed * np.exp(1j * src_phase)

        channel = np.fft.ifft2(mixed_fft).real
        mixed[:, :, ch] = channel

    mixed = np.clip(mixed, 0.0, 1.0)
    mixed = (mixed * 255.0).astype(np.uint8)

    return mixed


class PolypDB_DATASET(Dataset):
    def __init__(
        self,
        samples_path,
        size,
        transform=None,
        use_freqampmix=False,
        freqampmix_p=0.35,
        alpha_range=(0.15, 0.35),
        low_freq_ratio=0.10
    ):
        super().__init__()

        self.samples_path = samples_path
        self.transform = transform
        self.n_samples = len(samples_path)
        self.size = size

        self.use_freqampmix = use_freqampmix
        self.freqampmix_p = freqampmix_p
        self.alpha_range = alpha_range
        self.low_freq_ratio = low_freq_ratio

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

        if self.use_freqampmix and random.random() < self.freqampmix_p and self.n_samples > 1:
            ref_index = random.randint(0, self.n_samples - 1)

            if ref_index == index:
                ref_index = (ref_index + 1) % self.n_samples

            ref_image_path = self.samples_path[ref_index][0]
            ref_image = cv2.imread(ref_image_path, cv2.IMREAD_COLOR)

            if ref_image is not None:
                ref_image = cv2.resize(ref_image, self.size, interpolation=cv2.INTER_LINEAR)

                alpha = random.uniform(self.alpha_range[0], self.alpha_range[1])

                image = freq_amp_mix(
                    image=image,
                    reference_image=ref_image,
                    alpha=alpha,
                    low_freq_ratio=self.low_freq_ratio
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


def get_loss_part(out, key):
    if key in out:
        return out[key].item()

    aliases = {
        "loss_ugel": ["loss_soft_ugel", "loss_uncertainty_edge", "loss_edge"],
        "loss_focal_tversky": ["loss_ft", "loss_tversky"],
        "loss_band": ["band_loss"],
        "loss_final": ["loss_dice_bce", "loss_main"]
    }

    for alias in aliases.get(key, []):
        if alias in out:
            return out[alias].item()

    return 0.0


def train(model, loader, optimizer, device):
    model.train()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    epoch_loss_final = 0.0
    epoch_loss_ft = 0.0
    epoch_loss_band = 0.0
    epoch_loss_ugel = 0.0

    for x, y, modalities in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        sample = {
            "images": x,
            "masks": y,
            "modalities": modalities
        }

        out = model(sample)

        y_pred = out["prediction"]
        loss = out["loss"]

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        epoch_loss_final += get_loss_part(out, "loss_final")
        epoch_loss_ft += get_loss_part(out, "loss_focal_tversky")
        epoch_loss_band += get_loss_part(out, "loss_band")
        epoch_loss_ugel += get_loss_part(out, "loss_ugel")

        y_pred = torch.sigmoid(y_pred)

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

    loss_parts = {
        "loss_final": epoch_loss_final / n_batches,
        "loss_focal_tversky": epoch_loss_ft / n_batches,
        "loss_band": epoch_loss_band / n_batches,
        "loss_ugel": epoch_loss_ugel / n_batches
    }

    return (
        epoch_loss / n_batches,
        [
            epoch_jac / n_batches,
            epoch_f1 / n_batches,
            epoch_recall / n_batches,
            epoch_precision / n_batches
        ],
        loss_parts
    )


def evaluate(model, loader, device):
    model.eval()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    epoch_loss_final = 0.0
    epoch_loss_ft = 0.0
    epoch_loss_band = 0.0
    epoch_loss_ugel = 0.0

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

            y_pred = out["prediction"]
            loss = out["loss"]

            epoch_loss += loss.item()

            epoch_loss_final += get_loss_part(out, "loss_final")
            epoch_loss_ft += get_loss_part(out, "loss_focal_tversky")
            epoch_loss_band += get_loss_part(out, "loss_band")
            epoch_loss_ugel += get_loss_part(out, "loss_ugel")

            y_pred = torch.sigmoid(y_pred)

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

    loss_parts = {
        "loss_final": epoch_loss_final / n_batches,
        "loss_focal_tversky": epoch_loss_ft / n_batches,
        "loss_band": epoch_loss_band / n_batches,
        "loss_ugel": epoch_loss_ugel / n_batches
    }

    return (
        epoch_loss / n_batches,
        [
            epoch_jac / n_batches,
            epoch_f1 / n_batches,
            epoch_recall / n_batches,
            epoch_precision / n_batches
        ],
        loss_parts
    )


def run_experiment(path):
    model_name = "FocusNet"

    current_modality = infer_modality_from_path(path)

    experiment_name = f"FocusNet_DGFR_BandHead_UGEL_FreqAmpMix_modality_{current_modality}"

    create_dir("files")
    create_dir(f"files/modality_wise/{model_name}")

    train_log_path = (
        f"files/modality_wise/{model_name}/"
        f"train_log_{current_modality}_{EXPERIMENT_TAG}.txt"
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
        f"checkpoint_{current_modality}_{EXPERIMENT_TAG}.pth"
    )

    wandb.init(
    project="polyp-segmentation-focusnet",
    name=experiment_name,
    reinit="finish_previous",
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
        "train_path": path,
        "freqampmix_p": 0.35,
        "freqampmix_alpha_range": "0.15-0.35",
        "freqampmix_low_freq_ratio": 0.10,
        "loss": "Internal DiceBCE + FocalTversky + Band + UGEL",
        "augmentation": "Spatial aug + FreqAmpMix"
    }
)

    data_str = f"Experiment: {experiment_name}\n"
    data_str += f"Variant: {VARIANT_NAME}\n"
    data_str += f"Commit name: dgfr_bandhead_ugel_freqampmix\n"
    data_str += f"Setting: modality_wise\n"
    data_str += f"Modality: {current_modality}\n"
    data_str += f"Image Size: {size}\n"
    data_str += f"Batch Size: {batch_size}\n"
    data_str += f"LR: {lr}\n"
    data_str += f"Weight Decay: {weight_decay}\n"
    data_str += f"Epochs: {num_epochs}\n"
    data_str += f"Early Stopping Patience: {early_stopping_patience}\n"
    data_str += f"Path: {path}\n"
    data_str += f"FreqAmpMix: p=0.35, alpha=(0.15, 0.35), low_freq_ratio=0.10\n"

    print_and_save(train_log_path, data_str)

    train_samples, valid_samples, test_samples = load_polypdb_modality_data(path)
    train_samples = shuffle(train_samples, random_state=42)

    data_str = "Dataset Size:\n"
    data_str += f"Train: {len(train_samples)} - Valid: {len(valid_samples)} - Test: {len(test_samples)}\n"

    print_and_save(train_log_path, data_str)

    if len(train_samples) == 0 or len(valid_samples) == 0:
        print_and_save(
            train_log_path,
            f"Skipping {current_modality}: empty train or validation split.\n"
        )

        wandb.finish()
        return

    transform = A.Compose([
        A.Rotate(limit=35, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.CoarseDropout(
            max_holes=8,
            max_height=24,
            max_width=24,
            p=0.25
        )
    ])

    train_dataset = PolypDB_DATASET(
        samples_path=train_samples,
        size=size,
        transform=transform,
        use_freqampmix=True,
        freqampmix_p=0.35,
        alpha_range=(0.15, 0.35),
        low_freq_ratio=0.10
    )

    valid_dataset = PolypDB_DATASET(
        samples_path=valid_samples,
        size=size,
        transform=None,
        use_freqampmix=False
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

    print(f"train model: {model_name}")
    print(f"experiment: {experiment_name}")
    print(f"path: {path}")

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

    data_str = "Optimizer: AdamW\n"
    data_str += "Scheduler: ReduceLROnPlateau\n"
    data_str += "Loss: Internal DiceBCE + FocalTversky + Band + UGEL\n"
    data_str += "Important: FreqAmpMix is augmentation only, not an extra loss.\n"

    print_and_save(train_log_path, data_str)

    best_valid_f1 = 0.0
    early_stopping_count = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_metrics, train_loss_parts = train(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device
        )

        valid_loss, valid_metrics, valid_loss_parts = evaluate(
            model=model,
            loader=valid_loader,
            device=device
        )

        scheduler.step(valid_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        if valid_metrics[1] > best_valid_f1:
            data_str = (
                f"Valid F1 improved from {best_valid_f1:2.4f} "
                f"to {valid_metrics[1]:2.4f}. "
                f"Saving checkpoint: {checkpoint_path}"
            )

            print_and_save(train_log_path, data_str)

            best_valid_f1 = valid_metrics[1]
            torch.save(model.state_dict(), checkpoint_path)
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

            "train/loss_final": train_loss_parts["loss_final"],
            "train/loss_focal_tversky": train_loss_parts["loss_focal_tversky"],
            "train/loss_band": train_loss_parts["loss_band"],
            "train/loss_ugel": train_loss_parts["loss_ugel"],

            "valid/loss_final": valid_loss_parts["loss_final"],
            "valid/loss_focal_tversky": valid_loss_parts["loss_focal_tversky"],
            "valid/loss_band": valid_loss_parts["loss_band"],
            "valid/loss_ugel": valid_loss_parts["loss_ugel"],

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

        data_str += (
            f"\t Valid Loss Parts "
            f"- Final: {valid_loss_parts['loss_final']:.4f} "
            f"- FT: {valid_loss_parts['loss_focal_tversky']:.4f} "
            f"- Band: {valid_loss_parts['loss_band']:.4f} "
            f"- UGEL: {valid_loss_parts['loss_ugel']:.4f}\n"
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
    del optimizer
    del scheduler

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
