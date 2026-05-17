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


VARIANT_SLUG = "cross_expert_consensus_distillation_ema"
VARIANT_NAME = "DGFR+BandHead+CrossExpertConsensusDistillation+EMA"

EMA_DECAY = 0.995
EMA_CONSISTENCY_WEIGHT = 0.03
EMA_CONFIDENCE_THRESHOLD = 0.70

DISTILL_RAMP_EPOCHS = 30


TEACHER_FILE_RULES = {
    "UGEL": {
        "aliases": [
            "uncertainty_gated_edge_loss"
        ],
        "keyword_groups": [
            ["uncertainty", "gated", "edge", "loss"],
            ["ugel"]
        ]
    },
    "RAHS": {
        "aliases": [
            "region_adaptive_hybrid_supervision_ema"
        ],
        "keyword_groups": [
            ["region", "adaptive", "hybrid", "supervision", "ema"]
        ]
    },
    "WAVELET_EMA": {
        "aliases": [
            "wavelet_adaptive_ugel_ema"
        ],
        "keyword_groups": [
            ["wavelet", "adaptive", "ugel", "ema"]
        ]
    }
}


MODALITY_CONFIG = {
    "WLI": {
        "initial_teacher": "RAHS",
        "teachers": [
            ("RAHS", 0.70),
            ("WAVELET_EMA", 0.20),
            ("UGEL", 0.10)
        ],
        "soft_kd_weight": 0.18,
        "boundary_kd_weight": 0.08
    },
    "BLI": {
        "initial_teacher": "UGEL",
        "teachers": [
            ("UGEL", 0.85),
            ("RAHS", 0.15)
        ],
        "soft_kd_weight": 0.25,
        "boundary_kd_weight": 0.12
    },
    "FICE": {
        "initial_teacher": "RAHS",
        "teachers": [
            ("RAHS", 0.70),
            ("UGEL", 0.30)
        ],
        "soft_kd_weight": 0.22,
        "boundary_kd_weight": 0.10
    },
    "LCI": {
        "initial_teacher": "WAVELET_EMA",
        "teachers": [
            ("WAVELET_EMA", 0.70),
            ("UGEL", 0.30)
        ],
        "soft_kd_weight": 0.25,
        "boundary_kd_weight": 0.12
    },
    "NBI": {
        "initial_teacher": "UGEL",
        "teachers": [
            ("UGEL", 0.80),
            ("WAVELET_EMA", 0.20)
        ],
        "soft_kd_weight": 0.25,
        "boundary_kd_weight": 0.12
    }
}


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

    total_len = len(samples)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)

    train_samples = samples[:train_len]
    valid_samples = samples[train_len:train_len + val_len]
    test_samples = samples[train_len + val_len:]

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


def clean_state_dict(raw_state_dict):
    cleaned = {}

    for key, value in raw_state_dict.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        cleaned[new_key] = value

    return cleaned


def load_compatible_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    checkpoint = clean_state_dict(checkpoint)

    model_state = model.state_dict()

    compatible_state = {
        key: value
        for key, value in checkpoint.items()
        if key in model_state
        and model_state[key].shape == value.shape
    }

    model.load_state_dict(compatible_state, strict=False)

    print(
        f"Loaded compatible checkpoint: {checkpoint_path} | "
        f"Matched tensors: {len(compatible_state)}/{len(model_state)}"
    )


def resolve_teacher_checkpoint(directory, prefix, teacher_key):
    rule = TEACHER_FILE_RULES[teacher_key]

    for alias in rule["aliases"]:
        exact_path = os.path.join(
            directory,
            f"checkpoint_{prefix}_{alias}.pth"
        )

        if os.path.exists(exact_path):
            return exact_path

    candidate_files = sorted(
        glob(os.path.join(directory, f"checkpoint_{prefix}_*.pth"))
    )

    for candidate in candidate_files:
        basename = os.path.basename(candidate).lower()

        for keyword_group in rule["keyword_groups"]:
            if all(keyword in basename for keyword in keyword_group):
                return candidate

    return None


def load_teacher_models(
    device,
    teacher_directory,
    prefix,
    teacher_specs
):
    teacher_models = []
    teacher_weights = []
    teacher_names = []

    for teacher_key, teacher_weight in teacher_specs:
        checkpoint_path = resolve_teacher_checkpoint(
            directory=teacher_directory,
            prefix=prefix,
            teacher_key=teacher_key
        )

        if checkpoint_path is None:
            print(
                f"Teacher checkpoint not found for {teacher_key} "
                f"with prefix {prefix}. Skipping."
            )
            continue

        teacher_model = FocusNet().to(device)
        load_compatible_checkpoint(
            teacher_model,
            checkpoint_path,
            device
        )

        teacher_model.eval()

        for param in teacher_model.parameters():
            param.requires_grad = False

        teacher_models.append(teacher_model)
        teacher_weights.append(teacher_weight)
        teacher_names.append(teacher_key)

    if len(teacher_models) == 0:
        raise RuntimeError(
            f"No teacher checkpoints found for prefix: {prefix}"
        )

    teacher_weights = np.array(teacher_weights, dtype=np.float32)
    teacher_weights = teacher_weights / teacher_weights.sum()
    teacher_weights = teacher_weights.tolist()

    print("Teachers loaded:")
    for name, weight in zip(teacher_names, teacher_weights):
        print(f" - {name}: normalized weight {weight:.4f}")

    return teacher_models, teacher_weights, teacher_names


def initialize_student_from_teacher(
    student_model,
    ema_model,
    device,
    teacher_directory,
    prefix,
    teacher_key
):
    checkpoint_path = resolve_teacher_checkpoint(
        directory=teacher_directory,
        prefix=prefix,
        teacher_key=teacher_key
    )

    if checkpoint_path is None:
        print(
            f"Initial teacher checkpoint not found for {teacher_key}. "
            f"Student will start from pretrained FocusNet backbone."
        )
        return

    print(
        f"Initializing student and EMA student from teacher: "
        f"{teacher_key}"
    )

    load_compatible_checkpoint(
        student_model,
        checkpoint_path,
        device
    )

    load_compatible_checkpoint(
        ema_model,
        checkpoint_path,
        device
    )


def sobel_edge_map(x):
    device = x.device

    sobel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0],
          [-2.0, 0.0, 2.0],
          [-1.0, 0.0, 1.0]]],
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    sobel_y = torch.tensor(
        [[[-1.0, -2.0, -1.0],
          [0.0, 0.0, 0.0],
          [1.0, 2.0, 1.0]]],
        dtype=torch.float32,
        device=device
    ).unsqueeze(0)

    grad_x = F.conv2d(x, sobel_x, padding=1)
    grad_y = F.conv2d(x, sobel_y, padding=1)

    edge = torch.sqrt(
        grad_x.pow(2) + grad_y.pow(2) + 1e-6
    )

    return edge


def build_teacher_consensus(
    teacher_models,
    teacher_weights,
    sample
):
    teacher_predictions = []

    with torch.no_grad():
        for teacher_model in teacher_models:
            teacher_out = teacher_model(sample)
            teacher_prob = torch.sigmoid(teacher_out["prediction"])
            teacher_predictions.append(teacher_prob)

    stacked = torch.stack(teacher_predictions, dim=0)

    weights = torch.tensor(
        teacher_weights,
        dtype=stacked.dtype,
        device=stacked.device
    ).view(-1, 1, 1, 1, 1)

    consensus = (stacked * weights).sum(dim=0)

    if stacked.shape[0] > 1:
        disagreement = stacked.std(dim=0)
        agreement = torch.clamp(
            1.0 - disagreement / 0.50,
            min=0.0,
            max=1.0
        )
    else:
        agreement = torch.ones_like(consensus)

    certainty = torch.abs(consensus - 0.5) * 2.0

    confidence = torch.clamp(
        certainty * agreement,
        min=0.0,
        max=1.0
    )

    return consensus.detach(), confidence.detach()


def soft_consensus_distillation_loss(
    student_logits,
    teacher_consensus,
    confidence
):
    distill_map = F.binary_cross_entropy_with_logits(
        student_logits,
        teacher_consensus,
        reduction="none"
    )

    confidence_mask = (
        confidence >= 0.30
    ).float()

    pixel_weight = (
        0.25 + 0.75 * confidence
    ) * confidence_mask

    loss = (
        distill_map * pixel_weight
    ).sum() / (
        pixel_weight.sum() + 1e-6
    )

    return loss


def boundary_consensus_distillation_loss(
    student_logits,
    teacher_consensus,
    confidence
):
    student_prob = torch.sigmoid(student_logits)

    student_edge = sobel_edge_map(student_prob)
    teacher_edge = sobel_edge_map(teacher_consensus)

    edge_scale = teacher_edge / (
        teacher_edge.amax(dim=(2, 3), keepdim=True) + 1e-6
    )

    confidence_mask = (
        confidence >= 0.25
    ).float()

    pixel_weight = (
        0.25 + 0.75 * confidence
    ) * (
        1.0 + edge_scale
    ) * confidence_mask

    edge_difference = torch.abs(
        student_edge - teacher_edge
    )

    loss = (
        edge_difference * pixel_weight
    ).sum() / (
        pixel_weight.sum() + 1e-6
    )

    return loss


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

    loss = (
        consistency_map * confidence_mask
    ).sum() / (
        confidence_mask.sum() + 1e-6
    )

    return loss


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


def train_one_epoch(
    model,
    ema_model,
    teacher_models,
    teacher_weights,
    loader,
    optimizer,
    device,
    current_epoch,
    config
):
    model.train()
    ema_model.eval()

    epoch_loss = 0.0
    epoch_model_loss = 0.0
    epoch_soft_kd = 0.0
    epoch_boundary_kd = 0.0
    epoch_ema = 0.0

    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    distill_ramp = min(
        1.0,
        float(current_epoch + 1) / float(DISTILL_RAMP_EPOCHS)
    )

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

        teacher_consensus, teacher_confidence = build_teacher_consensus(
            teacher_models=teacher_models,
            teacher_weights=teacher_weights,
            sample=sample
        )

        loss_soft_kd = soft_consensus_distillation_loss(
            student_logits=out["prediction"],
            teacher_consensus=teacher_consensus,
            confidence=teacher_confidence
        )

        loss_boundary_kd = boundary_consensus_distillation_loss(
            student_logits=out["prediction"],
            teacher_consensus=teacher_consensus,
            confidence=teacher_confidence
        )

        with torch.no_grad():
            ema_out = ema_model(sample)

        loss_ema = confidence_filtered_ema_consistency_loss(
            student_logits=out["prediction"],
            teacher_logits=ema_out["prediction"],
            confidence_threshold=EMA_CONFIDENCE_THRESHOLD
        )

        model_loss = out["loss"]

        total_loss = (
            model_loss
            + distill_ramp * config["soft_kd_weight"] * loss_soft_kd
            + distill_ramp * config["boundary_kd_weight"] * loss_boundary_kd
            + EMA_CONSISTENCY_WEIGHT * loss_ema
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
        epoch_soft_kd += loss_soft_kd.item()
        epoch_boundary_kd += loss_boundary_kd.item()
        epoch_ema += loss_ema.item()

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

    losses = {
        "total_loss": epoch_loss / n_batches,
        "model_loss": epoch_model_loss / n_batches,
        "soft_kd": epoch_soft_kd / n_batches,
        "boundary_kd": epoch_boundary_kd / n_batches,
        "ema_consistency": epoch_ema / n_batches,
        "distill_ramp": distill_ramp
    }

    return epoch_loss / n_batches, metrics, losses


def evaluate(model, loader, device):
    model.eval()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

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

    return epoch_loss / n_batches, metrics


def run_experiment(path):
    model_name = "FocusNet"

    current_modality = infer_modality_from_path(path)
    config = MODALITY_CONFIG[current_modality]

    experiment_name = (
        f"FocusNet_DGFR_BandHead_"
        f"CrossExpertConsensusDistillation_EMA_"
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

    teacher_directory = f"files/modality_wise/{model_name}"
    teacher_prefix = current_modality

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
            "ema_consistency_weight": EMA_CONSISTENCY_WEIGHT,
            "ema_confidence_threshold": EMA_CONFIDENCE_THRESHOLD,
            "distill_ramp_epochs": DISTILL_RAMP_EPOCHS,
            "soft_kd_weight": config["soft_kd_weight"],
            "boundary_kd_weight": config["boundary_kd_weight"],
            "teachers": str(config["teachers"]),
            "initial_teacher": config["initial_teacher"]
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
    data_str += f"Initial Teacher: {config['initial_teacher']}\n"
    data_str += f"Teacher Specs: {config['teachers']}\n"
    data_str += f"Soft KD Weight: {config['soft_kd_weight']}\n"
    data_str += f"Boundary KD Weight: {config['boundary_kd_weight']}\n"
    data_str += f"Checkpoint: {checkpoint_path}\n"

    print_and_save(train_log_path, data_str)

    train_samples, valid_samples, test_samples = load_polypdb_modality_data(path)
    train_samples = shuffle(train_samples, random_state=42)

    data_str = (
        f"Dataset Size:\n"
        f"Train: {len(train_samples)} - "
        f"Valid: {len(valid_samples)} - "
        f"Test: {len(test_samples)}\n"
    )

    print_and_save(train_log_path, data_str)

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

    initialize_student_from_teacher(
        student_model=model,
        ema_model=ema_model,
        device=device,
        teacher_directory=teacher_directory,
        prefix=teacher_prefix,
        teacher_key=config["initial_teacher"]
    )

    ema_model.eval()

    for param in ema_model.parameters():
        param.requires_grad = False

    teacher_models, teacher_weights, teacher_names = load_teacher_models(
        device=device,
        teacher_directory=teacher_directory,
        prefix=teacher_prefix,
        teacher_specs=config["teachers"]
    )

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

        train_loss, train_metrics, train_losses = train_one_epoch(
            model=model,
            ema_model=ema_model,
            teacher_models=teacher_models,
            teacher_weights=teacher_weights,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            current_epoch=epoch,
            config=config
        )

        valid_loss, valid_metrics = evaluate(
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

            "train/model_loss": train_losses["model_loss"],
            "train/soft_kd": train_losses["soft_kd"],
            "train/boundary_kd": train_losses["boundary_kd"],
            "train/ema_consistency": train_losses["ema_consistency"],
            "train/distill_ramp": train_losses["distill_ramp"],

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
            f"\t Distillation "
            f"- Soft KD: {train_losses['soft_kd']:.4f} "
            f"- Boundary KD: {train_losses['boundary_kd']:.4f} "
            f"- EMA: {train_losses['ema_consistency']:.4f} "
            f"- Ramp: {train_losses['distill_ramp']:.4f}\n"
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

    for teacher_model in teacher_models:
        del teacher_model

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
