"""Haar-based face detection and fixed-size crop generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def sorted_image_paths(folder: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in suffixes)


def detect_face_crop(
    image: np.ndarray,
    crop_size: int = 128,
    scale1: float = 3.0,
    scale2: float = 3.0,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None, np.ndarray]:
    """Detect a face and return a fixed crop in the first-downsampled image.

    The preprocessing follows the experiment code:
    original image -> grayscale -> 3x downsample -> 3x downsample for Haar.
    Haar coordinates are mapped back to the first-downsampled image, and a
    fixed 128 x 128 crop is taken from the top-left corner of the detected box.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.resize(
        gray,
        (int(gray.shape[1] / scale1), int(gray.shape[0] / scale1)),
        interpolation=cv2.INTER_NEAREST,
    )
    gray2 = cv2.resize(
        gray1,
        (int(gray1.shape[1] / scale2), int(gray1.shape[0] / scale2)),
        interpolation=cv2.INTER_NEAREST,
    )

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray2, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None, None, gray1

    x, y, w, h = (faces[0] * int(scale2)).astype(int)
    x = max(0, min(x, gray1.shape[1] - crop_size))
    y = max(0, min(y, gray1.shape[0] - crop_size))
    crop = gray1[y : y + crop_size, x : x + crop_size]
    rect = (x, y, crop_size, crop_size)
    return crop, rect, gray1


def draw_detection(gray_image: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    output = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    x, y, w, h = rect
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return output


def crop_folder(input_dir: Path, output_dir: Path, overlay_dir: Path | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if overlay_dir is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    for image_path in sorted_image_paths(input_dir):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        crop, rect, gray1 = detect_face_crop(image)
        if crop is None or rect is None:
            print(f"no face: {image_path}")
            continue

        relative = image_path.relative_to(input_dir)
        crop_path = output_dir / relative.with_suffix(".png")
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(crop_path), crop)

        if overlay_dir is not None:
            overlay_path = overlay_dir / relative.with_suffix(".png")
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(overlay_path), draw_detection(gray1, rect))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 128 x 128 Haar face crops.")
    parser.add_argument("--input-dir", type=Path, default=Path("../data/raw_test"))
    parser.add_argument("--output-dir", type=Path, default=Path("../outputs/detected_crops"))
    parser.add_argument("--overlay-dir", type=Path, default=Path("../outputs/detection_overlays"))
    args = parser.parse_args()
    crop_folder(args.input_dir, args.output_dir, args.overlay_dir)


if __name__ == "__main__":
    main()
