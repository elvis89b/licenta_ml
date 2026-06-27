import os
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import torch

from model.FocusNet_UGEL import FocusNet
from lib import *
from utils import create_dir, seeding, calculate_metrics


def load_modality_test_data(path):
    images_jpg = sorted(glob(os.path.join(path, "images", "*.jpg")))
    images_png = sorted(glob(os.path.join(path, "images", "*.png")))
    images = images_jpg if len(images_jpg) > 0 else images_png

    image_names = [os.path.splitext(os.path.basename(file))[0] for file in images]
    samples = []

    for image_name in image_names:
        image_jpg = os.path.join(path, "images", f"{image_name}.jpg")
        image_png = os.path.join(path, "images", f"{image_name}.png")

        if os.path.exists(image_jpg):
            image = image_jpg
        elif os.path.exists(image_png):
            image = image_png
        else:
            continue

        mask_png = os.path.join(path, "masks", f"{image_name}.png")
        mask_jpg = os.path.join(path, "masks", f"{image_name}.jpg")

        if os.path.exists(mask_png):
            mask = mask_png
        elif os.path.exists(mask_jpg):
            mask = mask_jpg
        else:
            continue

        samples.append((image, mask))

    return samples


def split_wli_data(path):
    samples = load_modality_test_data(path)

    total_len = len(samples)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)

    train_samples = samples[:train_len]
    valid_samples = samples[train_len:train_len + val_len]
    test_samples = samples[train_len + val_len:]

    return train_samples, valid_samples, test_samples


def evaluate(model, save_path, test_samples, size, device):
    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []

    for _, (x, y) in tqdm(enumerate(test_samples), total=len(test_samples)):
        name = y.split("/")[-1].split(".")[0]

        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, size)
        save_img = image.copy()
        image = np.transpose(image, (2, 0, 1))
        image = image / 255.0
        image = np.expand_dims(image, axis=0).astype(np.float32)
        image = torch.from_numpy(image).to(device)

        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, size)
        save_mask = np.expand_dims(mask, axis=-1)
        save_mask = np.concatenate([save_mask, save_mask, save_mask], axis=2)

        mask = np.expand_dims(mask, axis=0)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=0).astype(np.float32)
        mask = torch.from_numpy(mask).to(device)

        with torch.no_grad():
            start_time = time.time()

            sample = {'images': image, 'masks': mask}
            out = model(sample)
            y_pred = out['prediction']

            end_time = time.time() - start_time
            time_taken.append(end_time)

            y_pred = torch.sigmoid(y_pred)
            score = calculate_metrics(mask, y_pred)
            metrics_score = list(map(add, metrics_score, score))

            y_pred = y_pred[0].cpu().numpy()
            y_pred = np.squeeze(y_pred, axis=0)
            y_pred = (y_pred > 0.5).astype(np.uint8) * 255
            y_pred = np.expand_dims(y_pred, axis=-1)
            y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=2)

        line = np.ones((size[0], 10, 3), dtype=np.uint8) * 255
        cat_images = np.concatenate([save_img, line, save_mask, line, y_pred], axis=1)

        cv2.imwrite(f"{save_path}/joint/{name}.jpg", cat_images)
        cv2.imwrite(f"{save_path}/mask/{name}.jpg", y_pred)

    jaccard = metrics_score[0] / len(test_samples)
    f1 = metrics_score[1] / len(test_samples)
    recall = metrics_score[2] / len(test_samples)
    precision = metrics_score[3] / len(test_samples)
    acc = metrics_score[4] / len(test_samples)
    f2 = metrics_score[5] / len(test_samples)

    print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f} - F2: {f2:1.4f}")

    mean_time_taken = np.mean(time_taken)
    mean_fps = 1 / mean_time_taken
    print("Mean FPS: ", mean_fps)


if __name__ == "__main__":
    seeding(42)

    model_name = 'FocusNet'
    checkpoint_path = f"files/modality_wise/{model_name}/checkpoint.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FocusNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"test model: {model_name}")

    test_modality_list = ['WLI', 'BLI', 'FICE', 'LCI', 'NBI']

    for test_modality in test_modality_list:
        save_path = f"files/modality_wise/{model_name}/results/{test_modality}"
        test_path = f"data/PolypDB/PolypDB_modality_wise/{test_modality}"

        if test_modality == 'WLI':
            _, _, test_samples = split_wli_data(test_path)
        else:
            test_samples = load_modality_test_data(test_path)

        print(f"test_modality: {test_modality}, test size: {len(test_samples)}")

        create_dir(save_path)
        create_dir(f"{save_path}/mask")
        create_dir(f"{save_path}/joint")

        size = (256, 256)
        evaluate(model, save_path, test_samples, size, device)
