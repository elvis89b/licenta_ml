import os
import random
import time
import datetime
import numpy as np
import albumentations as A
import cv2
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
import wandb

from utils import seeding, create_dir, print_and_save, epoch_time, calculate_metrics
from model.FocusNet import *
from metrics import DiceBCELoss
from sklearn.utils import shuffle
from lib import *


def load_names(path, file_path):
    f = open(file_path, "r")
    data = f.read().split("\n")[:-1]
    images = [os.path.join(path, "images", name) + ".jpg" for name in data]
    masks = [os.path.join(path, "masks", name) + ".jpg" for name in data]
    return images, masks


def load_kvasir_data(path):
    train_names_path = f"{path}/train.txt"
    valid_names_path = f"{path}/val.txt"

    train_x, train_y = load_names(path, train_names_path)
    valid_x, valid_y = load_names(path, valid_names_path)

    return (train_x, train_y), (valid_x, valid_y)


def load_test_data(path):
    images = sorted(glob(os.path.join(path, "images", "*.png")))
    image_names = [os.path.splitext(os.path.basename(file))[0] for file in images]
    samples = []
    for image_name in image_names:
        image = os.path.join(path, "images", f"{image_name}.png")
        mask = os.path.join(path, "masks", f"{image_name}.png")
        samples.append((image, mask))

    return samples


def load_polypdb_data_0(path):
    def get_data(path, name, modality):
        samples = []
        images = sorted(glob(os.path.join(path, name, modality, "images", "*.jpg")))
        image_names = [os.path.splitext(os.path.basename(file))[0] for file in images]

        mask_path = os.path.join(path, name, modality, "masks")
        for image_name in image_names:
            image = os.path.join(path, name, modality, "images", f"{image_name}.jpg")
            mask_jpg = os.path.join(mask_path, f"{image_name}.jpg")
            mask_png = os.path.join(mask_path, f"{image_name}.png")
            if os.path.exists(mask_jpg):
                mask = mask_jpg
            elif os.path.exists(mask_png):
                mask = mask_png
            else:
                continue
            samples.append((image, mask))
        return samples

    all_samples = []
    modality_list = {'BKAI': ['BLI', 'FICE', 'LCI', 'WLI'],
                     'Karolinska': ['WLI'],
                     'Simula': ['NBI', 'WLI']}
    for name in ['BKAI', 'Karolinska', 'Simula']:
        for modality in modality_list[name]:
            all_samples += get_data(path, name, modality)

    total_len = len(all_samples)
    train_len = int(0.8 * total_len)
    test_len = int(0.1 * total_len)
    val_len = total_len - train_len - test_len

    train_samples = all_samples[:train_len]
    test_samples = all_samples[train_len:train_len + test_len]
    valid_samples = all_samples[train_len + test_len:]

    return [train_samples, test_samples, valid_samples]


def load_polypdb_data(path):
    def get_data(path, name, modality):
        samples = []
        images = sorted(glob(os.path.join(path, name, modality, "images", "*.jpg")))
        image_names = [os.path.splitext(os.path.basename(file))[0] for file in images]

        mask_path = os.path.join(path, name, modality, "masks")
        for image_name in image_names:
            image = os.path.join(path, name, modality, "images", f"{image_name}.jpg")
            mask_jpg = os.path.join(mask_path, f"{image_name}.jpg")
            mask_png = os.path.join(mask_path, f"{image_name}.png")
            if os.path.exists(mask_jpg):
                mask = mask_jpg
            elif os.path.exists(mask_png):
                mask = mask_png
            else:
                continue
            samples.append((image, mask))
        return samples

    train_samples = []
    valid_samples = []
    test_samples = []
    modality_list = {'BKAI': ['BLI', 'FICE', 'LCI', 'WLI'],
                     'Karolinska': ['WLI'],
                     'Simula': ['NBI', 'WLI']}
    for name in ['BKAI', 'Karolinska', 'Simula']:
        for modality in modality_list[name]:
            modality_data = get_data(path, name, modality)
            modality_len = len(modality_data)
            modality_train_len = int(0.8 * modality_len)
            modality_val_len = int(0.1 * modality_len)
            train_samples += modality_data[:modality_train_len]
            valid_samples += modality_data[modality_train_len:modality_train_len + modality_val_len]
            test_samples += modality_data[modality_train_len + modality_val_len:]

    return [train_samples, valid_samples, test_samples]


def load_polypdb_wli_data(path):
    def get_data(path):
        samples = []
        images = sorted(glob(os.path.join(path, "images", "*.jpg")))
        image_names = [os.path.splitext(os.path.basename(file))[0] for file in images]

        for image_name in image_names:
            image = os.path.join(path, "images", f"{image_name}.jpg")
            mask_jpg = os.path.join(path, "masks", f"{image_name}.jpg")
            mask_png = os.path.join(path, "masks", f"{image_name}.png")
            if os.path.exists(mask_png):
                mask = mask_png
            elif os.path.exists(mask_jpg):
                mask = mask_jpg
            else:
                continue
            samples.append((image, mask))
        return samples

    modality_data = get_data(path)
    modality_len = len(modality_data)
    modality_train_len = int(0.8 * modality_len)
    modality_val_len = int(0.1 * modality_len)
    train_samples = modality_data[:modality_train_len]
    valid_samples = modality_data[modality_train_len:modality_train_len + modality_val_len]
    test_samples = modality_data[modality_train_len + modality_val_len:]

    return [train_samples, valid_samples, test_samples]


def global_intensity_nonlinear(image, p=0.5):
    if random.random() > p:
        return image

    image = image.astype(np.float32) / 255.0

    gamma = random.uniform(0.7, 1.5)
    gain = random.uniform(0.85, 1.15)

    image = gain * np.power(np.clip(image, 0.0, 1.0), gamma)
    image = np.clip(image, 0.0, 1.0)

    image = (image * 255.0).astype(np.uint8)
    return image


class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None, use_gin=False, gin_p=0.5):
        super().__init__()
        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.n_samples = len(images_path)
        self.size = size
        self.use_gin = use_gin
        self.gin_p = gin_p

    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)

        if self.use_gin:
            image = global_intensity_nonlinear(image, p=self.gin_p)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        image = cv2.resize(image, self.size)
        image = np.transpose(image, (2, 0, 1))
        image = image / 255.0

        mask = cv2.resize(mask, self.size)
        mask = np.expand_dims(mask, axis=0)
        mask = mask / 255.0

        return image, mask

    def __len__(self):
        return self.n_samples


class PolypDB_DATASET(Dataset):
    def __init__(self, samples_path, size, transform=None, use_gin=False, gin_p=0.5):
        super().__init__()
        self.samples_path = samples_path
        self.transform = transform
        self.n_samples = len(samples_path)
        self.size = size
        self.use_gin = use_gin
        self.gin_p = gin_p

    def __getitem__(self, index):
        image = cv2.imread(self.samples_path[index][0], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.samples_path[index][1], cv2.IMREAD_GRAYSCALE)

        if self.use_gin:
            image = global_intensity_nonlinear(image, p=self.gin_p)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        image = cv2.resize(image, self.size)
        image = np.transpose(image, (2, 0, 1))
        image = image / 255.0

        mask = cv2.resize(mask, self.size)
        mask = np.expand_dims(mask, axis=0)
        mask = mask / 255.0

        return image, mask

    def __len__(self):
        return self.n_samples


def train(model, loader, optimizer, loss_fn, device):
    model.train()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        sample = {'images': x, 'masks': y}
        out = model(sample)
        y_pred = out['prediction']
        loss = out['loss']

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        batch_jac, batch_f1, batch_recall, batch_precision = [], [], [], []

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

    epoch_loss /= len(loader)
    epoch_jac /= len(loader)
    epoch_f1 /= len(loader)
    epoch_recall /= len(loader)
    epoch_precision /= len(loader)

    return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]


def evaluate(model, loader, loss_fn, device):
    model.eval()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            sample = {'images': x, 'masks': y}
            out = model(sample)
            y_pred = out['prediction']
            loss = out['loss']

            epoch_loss += loss.item()

            batch_jac, batch_f1, batch_recall, batch_precision = [], [], [], []

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

    epoch_loss /= len(loader)
    epoch_jac /= len(loader)
    epoch_f1 /= len(loader)
    epoch_recall /= len(loader)
    epoch_precision /= len(loader)

    return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]


if __name__ == "__main__":
    seeding(42)
    create_dir("files")

    model_name = 'FocusNet'
    experiment_name = "FocusNet_DGFR_BandHead_GIN_center"

    train_log_path = f"files/center_wise/{model_name}/train_log.txt"
    if os.path.exists(train_log_path):
        print("Log file exists")
    else:
        create_dir(f"files/center_wise/{model_name}")
        with open(train_log_path, "w") as train_log:
            train_log.write("\n")

    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)
    print("")

    image_size = 256
    size = (image_size, image_size)
    batch_size = 16
    num_epochs = 500
    lr = 1e-4
    early_stopping_patience = 50
    checkpoint_path = f"files/center_wise/{model_name}/checkpoint.pth"
    path = "data/PolypDB/PolypDB_center_wise/Simula/WLI"

    use_gin = True
    gin_p = 0.5

    wandb.init(
        project="polyp-segmentation-focusnet",
        name=experiment_name,
        config={
            "model": model_name,
            "variant": "DGFR+BandHead+GIN",
            "setting": "center_wise",
            "image_size": image_size,
            "batch_size": batch_size,
            "epochs": num_epochs,
            "lr": lr,
            "early_stopping_patience": early_stopping_patience,
            "train_path": path,
            "use_gin": use_gin,
            "gin_p": gin_p
        }
    )

    data_str = f"Image Size: {size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    data_str += f"Early Stopping Patience: {early_stopping_patience}\n"
    data_str += f"Use GIN: {use_gin}\nGIN p: {gin_p}\n"
    print_and_save(train_log_path, data_str)

    train_samples, valid_samples, test_samples = load_polypdb_wli_data(path)
    _ = shuffle(train_samples, random_state=42)

    data_str = f"Dataset Size:\nTrain: {len(train_samples)} - Valid: {len(valid_samples)} - Test: {len(test_samples)}\n"
    print_and_save(train_log_path, data_str)

    transform = A.Compose([
        A.Rotate(limit=35, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
    ])

    train_dataset = PolypDB_DATASET(train_samples, size, transform=transform, use_gin=use_gin, gin_p=gin_p)
    valid_dataset = PolypDB_DATASET(valid_samples, size, transform=None, use_gin=False, gin_p=0.0)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = eval(model_name)().to(device)
    print(f"train model: {model_name}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceBCELoss()
    loss_name = "BCE Dice Loss"

    data_str = f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    best_valid_metrics = 0.0
    early_stopping_count = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_metrics = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss, valid_metrics = evaluate(model, valid_loader, loss_fn, device)
        scheduler.step(valid_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        if valid_metrics[1] > best_valid_metrics:
            data_str = f"Valid F1 improved from {best_valid_metrics:2.4f} to {valid_metrics[1]:2.4f}. Saving checkpoint: {checkpoint_path}"
            print_and_save(train_log_path, data_str)

            best_valid_metrics = valid_metrics[1]
            torch.save(model.state_dict(), checkpoint_path)
            early_stopping_count = 0

        elif valid_metrics[1] < best_valid_metrics:
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
            "best_valid_f1": best_valid_metrics
        })

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tTrain Loss: {train_loss:.4f} - Jaccard: {train_metrics[0]:.4f} - F1: {train_metrics[1]:.4f} - Recall: {train_metrics[2]:.4f} - Precision: {train_metrics[3]:.4f}\n"
        data_str += f"\t Val. Loss: {valid_loss:.4f} - Jaccard: {valid_metrics[0]:.4f} - F1: {valid_metrics[1]:.4f} - Recall: {valid_metrics[2]:.4f} - Precision: {valid_metrics[3]:.4f}\n"
        print_and_save(train_log_path, data_str)

        if early_stopping_count == early_stopping_patience:
            data_str = f"Early stopping: validation loss stops improving from last {early_stopping_patience} continously.\n"
            print_and_save(train_log_path, data_str)
            break

    wandb.finish()
