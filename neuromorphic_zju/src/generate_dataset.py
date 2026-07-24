"""Generate noisy 4 x 4 Z/J/U samples from ideal binary letter patterns."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zju_utils import CLASS_NAMES, IDEAL_PATTERNS


OUTPUT = Path("data/zju_generated_10db_dataset.npz")


def add_orthogonal_noise(pattern: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Noise model used by the later MATLAB script zju_snr.m."""

    x = pattern.reshape(16)
    signal_power = np.sum(x**2)
    noise_power = signal_power * 10 ** (-snr_db / 10)
    noise = rng.random(16) * 2 - 1
    projection = np.dot(x, noise) / signal_power * x
    residual = noise - projection
    scaled_noise = residual / np.sqrt(np.sum(residual**2)) * np.sqrt(noise_power)
    sample = x + scaled_noise
    out_of_range = (sample < 0) | (sample > 1)
    sample[out_of_range] = x[out_of_range] - scaled_noise[out_of_range]
    return sample.reshape(4, 4)


def main() -> None:
    rng = np.random.default_rng(0)
    samples = []
    labels = []
    for label, name in enumerate(CLASS_NAMES):
        pattern = IDEAL_PATTERNS[str(name)]
        for _ in range(500):
            samples.append(add_orthogonal_noise(pattern, snr_db=10, rng=rng))
            labels.append(label)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT, X=np.stack(samples), y=np.array(labels), class_names=CLASS_NAMES)
    print(f"saved: {OUTPUT}")


if __name__ == "__main__":
    main()
