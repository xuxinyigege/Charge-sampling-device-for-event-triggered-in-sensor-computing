"""Small 4 x 4 array convolution demo for Z/J/U patterns.

The legacy files mainly document the fully connected 16 -> 8 first layer. This
script is included as a compact companion demo showing how a 4 x 4 GCCD array
can also be scanned by small convolution kernels.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from zju_utils import IDEAL_PATTERNS


OUTPUT = Path("outputs/convolution_demo.csv")


KERNELS = {
    "horizontal": np.array([[1, 1], [-1, -1]], dtype=float),
    "vertical": np.array([[1, -1], [1, -1]], dtype=float),
    "diagonal": np.array([[1, 0], [0, -1]], dtype=float),
}


def conv2_valid(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    height, width = image.shape
    kh, kw = kernel.shape
    output = np.zeros((height - kh + 1, width - kw + 1))
    for row in range(output.shape[0]):
        for col in range(output.shape[1]):
            output[row, col] = np.sum(image[row : row + kh, col : col + kw] * kernel)
    return output


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["letter", "kernel", "row", "col", "response"])
        for letter, pattern in IDEAL_PATTERNS.items():
            for kernel_name, kernel in KERNELS.items():
                response = conv2_valid(pattern, kernel)
                for row in range(response.shape[0]):
                    for col in range(response.shape[1]):
                        writer.writerow([letter, kernel_name, row, col, response[row, col]])
    print(f"saved: {OUTPUT}")


if __name__ == "__main__":
    main()
