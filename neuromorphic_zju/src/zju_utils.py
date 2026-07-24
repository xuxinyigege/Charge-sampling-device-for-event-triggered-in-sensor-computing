"""Utilities for the 4 x 4 Z/J/U neuromorphic classification experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np


CLASS_NAMES = np.array(["Z", "J", "U"])

IDEAL_PATTERNS = {
    "Z": np.array([[1, 1, 1, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 1, 1, 1]], dtype=float),
    "J": np.array([[1, 1, 1, 1], [0, 0, 1, 0], [1, 0, 1, 0], [1, 1, 1, 0]], dtype=float),
    "U": np.array([[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]], dtype=float),
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def train_test_split_by_class(
    X: np.ndarray,
    y: np.ndarray,
    train_per_class: int = 350,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Match the legacy MATLAB split: first 70% per class for training."""

    train_indices = []
    test_indices = []
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        train_indices.extend(indices[:train_per_class])
        test_indices.extend(indices[train_per_class:])
    return X[train_indices], y[train_indices], X[test_indices], y[test_indices]


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["X"].astype(float), data["y"].astype(int)


def flatten_patterns(X: np.ndarray) -> np.ndarray:
    """Flatten 4 x 4 patterns using MATLAB-compatible column-major order."""

    return X.reshape((X.shape[0], 16), order="F")


def mlp_forward(X_flat: np.ndarray, weights: dict[str, np.ndarray], round_first_layer: bool = False) -> np.ndarray:
    """Run the legacy 16 -> 8 -> 3 MLP.

    The first matrix multiplication corresponds to the experimentally relevant
    4 x 4 GCCD-array weighted-sum operation.
    """

    W1 = weights["W1"]
    if round_first_layer:
        W1 = np.round(W1)
    hidden_current = X_flat @ W1.T + weights["B1"]
    hidden_activation = sigmoid(hidden_current)
    logits = hidden_activation @ weights["W2"].T + weights["B2"]
    return sigmoid(logits)


def load_weights(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {name: data[name].astype(float) for name in ["W1", "W2", "B1", "B2"]}


def accuracy(probabilities: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(probabilities.argmax(axis=1) == labels))
