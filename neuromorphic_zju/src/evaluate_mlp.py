"""Evaluate the legacy MLP weights on noisy 4 x 4 Z/J/U samples."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from zju_utils import CLASS_NAMES, accuracy, flatten_patterns, load_dataset, load_weights, mlp_forward, train_test_split_by_class


DATASET = Path("data/zju_10db_dataset.npz")
WEIGHTS = Path("models/legacy_mlp_weights.npz")
OUTPUT_DIR = Path("outputs")


def main() -> None:
    X, y = load_dataset(DATASET)
    _, _, X_test, y_test = train_test_split_by_class(X, y)
    X_test_flat = flatten_patterns(X_test)
    weights = load_weights(WEIGHTS)

    probabilities = mlp_forward(X_test_flat, weights)
    probabilities_rounded = mlp_forward(X_test_flat, weights, round_first_layer=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "zju_predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "label", "prediction", "prob_Z", "prob_J", "prob_U"])
        for i, (label, probs) in enumerate(zip(y_test, probabilities)):
            writer.writerow([i, CLASS_NAMES[label], CLASS_NAMES[int(np.argmax(probs))], *probs])

    summary = {
        "task": "Z/J/U recognition from noisy 4x4 patterns",
        "samples_per_class": 500,
        "test_samples_per_class": 150,
        "snr_db": 10,
        "accuracy_float_first_layer": accuracy(probabilities, y_test),
        "accuracy_rounded_first_layer": accuracy(probabilities_rounded, y_test),
    }
    (OUTPUT_DIR / "zju_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
