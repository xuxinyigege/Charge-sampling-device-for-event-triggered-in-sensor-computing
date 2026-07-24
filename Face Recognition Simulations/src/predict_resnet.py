"""Evaluate the trained ResNet on cropped face images."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import load_model


def image_to_tensor(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("L")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)


def evaluate(test_dir: Path, weights: Path, output_csv: Path, num_classes: int) -> None:
    model = load_model(str(weights), num_classes=num_classes)
    rows = []
    correct = 0

    for class_dir in sorted(path for path in test_dir.iterdir() if path.is_dir()):
        label = int(class_dir.name)
        for image_path in sorted(class_dir.glob("*.png")):
            with torch.no_grad():
                logits = model(image_to_tensor(image_path))
                prediction = int(logits.argmax(1).item())
            correct += int(prediction == label)
            rows.append(
                {
                    "image": str(image_path),
                    "label": label,
                    "prediction": prediction,
                    "correct": prediction == label,
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "label", "prediction", "correct"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"accuracy: {correct}/{len(rows)} = {correct / len(rows):.4f}")
    print(f"saved: {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict labels for cropped face images.")
    parser.add_argument("--test-dir", type=Path, default=Path("../data/test_cropped"))
    parser.add_argument("--weights", type=Path, default=Path("../models/face_resnet_state.pt"))
    parser.add_argument("--output-csv", type=Path, default=Path("../outputs/predictions.csv"))
    parser.add_argument("--num-classes", type=int, default=10)
    args = parser.parse_args()
    evaluate(args.test_dir, args.weights, args.output_csv, args.num_classes)


if __name__ == "__main__":
    main()
