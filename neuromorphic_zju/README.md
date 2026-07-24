# 4 x 4 GCCD Array Z/J/U Recognition

This folder documents the legacy 4 x 4 GCCD-array experiment for recognizing
the letters `Z`, `J`, and `U`. The original experiment used noisy 4 x 4 binary
letter patterns and a small multilayer perceptron (MLP).

## What Was in the Legacy Folder

The legacy MATLAB files indicate the following workflow:

1. Define ideal `4 x 4` binary patterns for `Z`, `J`, and `U`.
2. Generate noisy samples at a target SNR, especially 10 dB.
3. Flatten each `4 x 4` pattern into 16 pixels.
4. Train an MLP with architecture `16 -> 8 -> 3`.
5. Treat the first matrix multiplication, `W1 * x`, as the operation that can
   be mapped to the 4 x 4 GCCD array.
6. Feed the 8 first-layer outputs into the remaining neural-network classifier.

The original folder also contained thousands of rendered PNG stimulus frames and
older MNIST/MLP reference files. Those are not included here because they are
large and can be regenerated.

## Contents

```text
neuromorphic_zju/
├── data/
│   ├── patterns.csv
│   └── zju_10db_dataset.npz
├── models/
│   └── legacy_mlp_weights.npz
├── outputs/
│   ├── zju_predictions.csv
│   ├── zju_summary.json
│   └── convolution_demo.csv
└── src/
    ├── zju_utils.py
    ├── generate_dataset.py
    ├── evaluate_mlp.py
    └── convolution_demo.py
```

## Data

- `zju_10db_dataset.npz` contains 1500 samples with shape `(1500, 4, 4)`.
- There are 500 noisy samples per class.
- Class order is `Z = 0`, `J = 1`, `U = 2`.
- The train/test split follows the MATLAB scripts: the first 350 samples per
  class are training data, and the remaining 150 samples per class are test data.

## Reproduce the MLP Evaluation

Run from this folder:

```bash
python src/evaluate_mlp.py
```

Expected output:

```text
accuracy_float_first_layer: 0.9933333333333333
accuracy_rounded_first_layer: 0.9955555555555555
```

The rounded-first-layer result emulates the legacy `simulation_bp.m` idea, where
the first-layer weights are rounded before the first weighted-sum stage.

## Convolution Demo

No standalone legacy convolution script was found in the old folder. A compact
4 x 4 valid-convolution example is included as an auxiliary demo:

```bash
python src/convolution_demo.py
```

It writes kernel responses for three simple 2 x 2 kernels to
`outputs/convolution_demo.csv`.

## Notes for GitHub Release

This release keeps the core data and model weights small and reproducible. The
large generated PNG stimulus folders, paper PDFs, and unrelated MNIST examples
from the old working directory are intentionally excluded.
