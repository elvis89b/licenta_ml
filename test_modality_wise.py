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


VARIANT_SLUG = "wavelet_adaptive_ugel_ema"

THRESHOLD_CANDIDATES = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75
]

EVAL_PROTOCOL = "paper"
# "paper": WLI uses split test, BLI/FICE/LCI/NBI use all samples, like your previous modality-wise scripts.
# "strict": all modalities use the 80/10/10 test split.


def load_modality_data(path):
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


def prepare_sample(image_path, mask_path, size, device):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    save_img = image.copy()

    image = np.transpose(image, (2, 0, 1))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image).to(device, dtype=torch.float32)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")

    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

    save_mask = np.expand_dims(mask, axis=-1)
    save_mask = np.concatenate([save_mask, save_mask, save_mask], axis=2)

    mask = np.expand_dims(mask, axis=0)
    mask = mask.astype(np.float32) / 255.0
    mask = (mask > 0.5).astype(np.float32)
    mask = np.expand_dims(mask, axis=0)
    mask = torch.from_numpy(mask).to(device, dtype=torch.float32)

    return image, mask, save_img, save_mask


def get_prediction(model, image, mask, modality):
    sample = {
        "images": image,
        "masks": mask,
        "modalities": [modality]
    }

    out = model(sample)
    y_pred = torch.sigmoid(out["prediction"])

    return y_pred


def calculate_metrics_with_threshold(mask, y_pred_prob, threshold):
    y_pred_bin = (y_pred_prob > threshold).float()
    return calculate_metrics(mask, y_pred_bin)


def calibrate_threshold(model, device, valid_samples, size, modality):
    if len(valid_samples) == 0:
        print("Validation split is empty. Using default threshold 0.50.")
        return 0.50

    threshold_scores = {}
    model.eval()

    with torch.no_grad():
        for threshold in THRESHOLD_CANDIDATES:
            metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            for image_path, mask_path in tqdm(
                valid_samples,
                total=len(valid_samples),
                desc=f"Calibrating threshold {threshold:.2f}"
            ):
                image, mask, _, _ = prepare_sample(
                    image_path=image_path,
                    mask_path=mask_path,
                    size=size,
                    device=device
                )

                y_pred = get_prediction(
                    model=model,
                    image=image,
                    mask=mask,
                    modality=modality
                )

                score = calculate_metrics_with_threshold(
                    mask=mask,
                    y_pred_prob=y_pred,
                    threshold=threshold
                )

                metrics_score = list(map(add, metrics_score, score))

            jaccard = metrics_score[0] / len(valid_samples)
            f1 = metrics_score[1] / len(valid_samples)
            recall = metrics_score[2] / len(valid_samples)
            precision = metrics_score[3] / len(valid_samples)
            f2 = metrics_score[5] / len(valid_samples)

            threshold_scores[threshold] = {
                "jaccard": jaccard,
                "f1": f1,
                "recall": recall,
                "precision": precision,
                "f2": f2
            }

            print(
                f"Threshold {threshold:.2f} | "
                f"Jaccard: {jaccard:.4f} - "
                f"F1: {f1:.4f} - "
                f"Recall: {recall:.4f} - "
                f"Precision: {precision:.4f} - "
                f"F2: {f2:.4f}"
            )

    best_threshold = max(
        threshold_scores.keys(),
        key=lambda t: (
            threshold_scores[t]["f1"],
            threshold_scores[t]["jaccard"],
            threshold_scores[t]["precision"],
            threshold_scores[t]["f2"]
        )
    )

    print(f"Selected threshold: {best_threshold:.2f}")

    return best_threshold


def evaluate(model, device, save_path, test_samples, size, modality, threshold):
    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []

    for _, (x, y) in tqdm(enumerate(test_samples), total=len(test_samples)):
        name = y.split("/")[-1].split(".")[0]

        image, mask, save_img, save_mask = prepare_sample(
            image_path=x,
            mask_path=y,
            size=size,
            device=device
        )

        with torch.no_grad():
            start_time = time.time()

            y_pred_prob = get_prediction(
                model=model,
                image=image,
                mask=mask,
                modality=modality
            )

            end_time = time.time() - start_time
            time_taken.append(end_time)

            score = calculate_metrics_with_threshold(
                mask=mask,
                y_pred_prob=y_pred_prob,
                threshold=threshold
            )

            metrics_score = list(map(add, metrics_score, score))

            y_pred = y_pred_prob[0].cpu().numpy()
            y_pred = np.squeeze(y_pred, axis=0)
            y_pred = y_pred > threshold
            y_pred = y_pred.astype(np.int32)
            y_pred = y_pred * 255
            y_pred = np.array(y_pred, dtype=np.uint8)
            y_pred = np.expand_dims(y_pred, axis=-1)
            y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=2)

        line = np.ones((size[0], 10, 3), dtype=np.uint8) * 255

        cat_images = np.concatenate(
            [save_img, line, save_mask, line, y_pred],
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
        f"Threshold: {threshold:.2f} - "
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
        "fps": mean_fps,
        "threshold": threshold
    }


if __name__ == "__main__":
    seeding(42)

    model_name = "FocusNet"
    size = (256, 256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_modality_list = ["WLI", "BLI", "FICE", "LCI", "NBI"]
    summary_results = {}

    for modality in test_modality_list:
        print("\n" + "=" * 100)
        print(f"Testing modality: {modality}")
        print("=" * 100 + "\n")

        checkpoint_path = (
            f"files/modality_wise/{model_name}/"
            f"checkpoint_{modality}_{VARIANT_SLUG}.pth"
        )

        if not os.path.exists(checkpoint_path):
            print(f"Skipping {modality}: checkpoint not found: {checkpoint_path}")
            continue

        test_path = f"data/PolypDB/PolypDB_modality_wise/{modality}"

        if not os.path.exists(test_path):
            print(f"Skipping {modality}: test path not found: {test_path}")
            continue

        _, valid_samples, split_test_samples = split_data(test_path)

        if EVAL_PROTOCOL == "paper":
            if modality == "WLI":
                test_samples = split_test_samples
            else:
                test_samples = load_modality_data(test_path)
        else:
            test_samples = split_test_samples

        print(f"Checkpoint: {checkpoint_path}")
        print(f"Test path: {test_path}")
        print(f"Validation size: {len(valid_samples)}")
        print(f"Test size: {len(test_samples)}")
        print(f"Eval protocol: {EVAL_PROTOCOL}")

        if len(test_samples) == 0:
            print(f"Skipping {modality}: empty test split.")
            continue

        save_path = (
            f"files/modality_wise/{model_name}/"
            f"results_{VARIANT_SLUG}/{modality}"
        )

        create_dir(save_path)
        create_dir(f"{save_path}/mask")
        create_dir(f"{save_path}/joint")

        model = FocusNet().to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        print(f"test model: {model_name}")

        threshold = calibrate_threshold(
            model=model,
            device=device,
            valid_samples=valid_samples,
            size=size,
            modality=modality
        )

        result = evaluate(
            model=model,
            device=device,
            save_path=save_path,
            test_samples=test_samples,
            size=size,
            modality=modality,
            threshold=threshold
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
            f"Threshold: {result['threshold']:.2f} - "
            f"Jaccard: {result['jaccard']:.4f} - "
            f"F1: {result['f1']:.4f} - "
            f"Recall: {result['recall']:.4f} - "
            f"Precision: {result['precision']:.4f} - "
            f"Acc: {result['accuracy']:.4f} - "
            f"F2: {result['f2']:.4f} - "
            f"FPS: {result['fps']:.2f}"
        )
