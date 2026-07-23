"""Run detection, recognition, and first-layer feature-map export for one image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from face_detection import detect_face_crop, draw_detection
from model import load_model


def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    array -= array.min()
    if array.max() > 1e-12:
        array /= array.max()
    return (255 * array).astype(np.uint8)


def run_demo(image_path: Path, weights: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    crop, rect, gray1 = detect_face_crop(image)
    if crop is None or rect is None:
        raise RuntimeError(f"No face detected in {image_path}")

    cv2.imwrite(str(output_dir / "gray_downsampled.png"), gray1)
    cv2.imwrite(str(output_dir / "face_detection.png"), draw_detection(gray1, rect))
    cv2.imwrite(str(output_dir / "face_crop.png"), crop)

    model = load_model(str(weights))
    x = torch.from_numpy(crop.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        stem_feature = model.stem(x)
        logits = model.forward_after_stem(stem_feature)

    feature_maps = stem_feature.squeeze(0).numpy()
    for index, feature_map in enumerate(feature_maps):
        image_map = normalize_to_uint8(feature_map)
        image_map = cv2.resize(image_map, (168, 168), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(output_dir / f"feature_map_{index}.png"), image_map)

    print(f"prediction: {int(logits.argmax(1).item())}")
    print(f"saved: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single-image face experiment demo.")
    parser.add_argument("--image", type=Path, default=Path("../data/raw_test/文件0.png"))
    parser.add_argument("--weights", type=Path, default=Path("../models/face_resnet_state.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("../outputs/demo"))
    args = parser.parse_args()
    run_demo(args.image, args.weights, args.output_dir)


if __name__ == "__main__":
    main()
