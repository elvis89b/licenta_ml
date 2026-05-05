import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from operator import add
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

import torch

from model.FocusNet import FocusNet
from utils import create_dir, seeding, calculate_metrics


EXPERIMENT_TAG = "ugel_freqampmix"


def load_modality_data(path):
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

    return samples


def split_data(path):
    samples = load_modality_data(path)

    total_len = len(samples)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)

    train_samples = samples[:train_len]
    valid_samples = samples[train_len:train_len + val_len]
    test_samples = samples[train_len + val_len:]

    return train_samples, valid_samples, test_samples


def evaluate(model, device, save_path, test_samples, size, modality):
    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []

    for _, (x, y) in tqdm(enumerate(test_samples), total=len(test_samples)):
        name = y.split("/")[-1].split(".")[0]

        image = cv2.imread(x, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Could not read image: {x}")

        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        save_img = image.copy()

        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)

        image = torch.from_numpy(image).to(device, dtype=torch.float32)

        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise ValueError(f"Could not read mask: {y}")

        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

        save_mask = np.expand_dims(mask, axis=-1)
        save_mask = np.concatenate([save_mask, save_mask, save_mask], axis=2)

        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        mask = torch.from_numpy(mask).to(device, dtype=torch.float32)

        with torch.no_grad():
            start_time = time.time()

            sample = {
                "images": image,
                "masks": mask,
                "modalities": [modality]
            }

            out = model(sample)
            y_pred = out["prediction"]

            end_time = time.time() - start_time
            time_taken.append(end_time)

            y_pred = torch.sigmoid(y_pred)

            score = calculate_metrics(mask, y_pred)
            metrics_score = list(map(add, metrics_score, score))

            y_pred = y_pred[0].cpu().numpy()
            y_pred = np.squeeze(y_pred, axis=0)
            y_pred = y_pred > 0.5
            y_pred = y_pred.astype(np.int32)
            y_pred = y_pred * 255
            y_pred = np.array(y_pred, dtype=np.uint8)
            y_pred = np.expand_dims(y_pred, axis=-1)
            y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=2)

        line = np.ones((size[0], 10, 3), dtype=np.uint8) * 255

        cat_images = np.concatenate(
            [
                save_img,
                line,
                save_mask,
                line,
                y_pred
            ],
            axis=1
        )

        cv2.imwrite(f"{save_path}/joint/{name}.jpg", cat_images)
        cv2.imwrite(f"{save_path}/mask/{name}.jpg", y_pred)

    jaccard = metrics_score[0] / len(test_samples)
    f1 = metrics_score[1] / len(test_samples)
    recall = metrics_score[2] / len(test_samples)
    precision = metrics_score[3] / len(test_samples)
    acc = metrics_score[4] / len(test_samples)
    f2 = metrics_score[5] / len(test_samples)

    print(
        f"Jaccard: {jaccard:1.4f} "
        f"- F1: {f1:1.4f} "
        f"- Recall: {recall:1.4f} "
        f"- Precision: {precision:1.4f} "
        f"- Acc: {acc:1.4f} "
        f"- F2: {f2:1.4f}"
    )

    mean_time_taken = np.mean(time_taken)
    mean_fps = 1 / mean_time_taken

    print("Mean FPS: ", mean_fps)

    return {
        "jaccard": jaccard,
        "f1": f1,
        "recall": recall,
        "precision": precision,
        "accuracy": acc,
        "f2": f2,
        "fps": mean_fps
    }


if __name__ == "__main__":
    seeding(42)

    model_name = "FocusNet"
    size = (256, 256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_modality_list = [
        "WLI",
        "BLI",
        "FICE",
        "LCI",
        "NBI"
    ]

    summary_results = {}

    for modality in test_modality_list:
        print("\n" + "=" * 100)
        print(f"Testing modality: {modality}")
        print("=" * 100 + "\n")

        checkpoint_path = (
            f"files/modality_wise/{model_name}/"
            f"checkpoint_{modality}_{EXPERIMENT_TAG}.pth"
        )

        if not os.path.exists(checkpoint_path):
            print(f"Skipping {modality}: checkpoint not found: {checkpoint_path}")
            continue

        test_path = f"data/PolypDB/PolypDB_modality_wise/{modality}"

        if not os.path.exists(test_path):
            print(f"Skipping {modality}: test path not found: {test_path}")
            continue

        if modality == "WLI":
            _, _, test_samples = split_data(test_path)
        else:
            test_samples = load_modality_data(test_path)

        print(f"Checkpoint: {checkpoint_path}")
        print(f"Test path: {test_path}")
        print(f"Test size: {len(test_samples)}")

        if len(test_samples) == 0:
            print(f"Skipping {modality}: empty test split.")
            continue

        save_path = (
            f"files/modality_wise/{model_name}/"
            f"results_{EXPERIMENT_TAG}/{modality}"
        )

        create_dir(save_path)
        create_dir(f"{save_path}/mask")
        create_dir(f"{save_path}/joint")

        model = FocusNet().to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        print(f"test model: {model_name}")

        result = evaluate(
            model=model,
            device=device,
            save_path=save_path,
            test_samples=test_samples,
            size=size,
            modality=modality
        )

        summary_results[modality] = result

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 100)
    print("MODALITY-WISE SUMMARY")
    print("=" * 100)

    for modality, result in summary_results.items():
        print(
            f"{modality} | "
            f"Jaccard: {result['jaccard']:.4f} - "
            f"F1: {result['f1']:.4f} - "
            f"Recall: {result['recall']:.4f} - "
            f"Precision: {result['precision']:.4f} - "
            f"Acc: {result['accuracy']:.4f} - "
            f"F2: {result['f2']:.4f} - "
            f"FPS: {result['fps']:.2f}"
        )
