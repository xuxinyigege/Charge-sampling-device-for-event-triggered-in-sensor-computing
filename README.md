# Charge sampling device for event-triggered in-sensor computing

This repository contains cleaned, publication-oriented code for two experiments
used in the GCCD in-sensor-computing study:

1. **Face Recognition Simulations**: Haar-based face detection and ResNet-based
   face recognition.
2. **In-sensor Computing Experiments**: 4 x 4 GCCD-array simulations for `Z`,
   `J`, and `U` letter recognition using fully connected and convolution-style
   operations.

Human-face image files are intentionally excluded from the public repository for
privacy and consent reasons. The directory structure is kept so users can place
their own authorized images in the expected locations.

## Repository Structure

```text
.
├── Face Recognition Simulations/
│   ├── data/
│   │   ├── raw_test/          # Put authorized raw test images here
│   │   ├── train_cropped/     # Put 128 x 128 training face crops here
│   │   └── test_cropped/      # Put 128 x 128 evaluation face crops here
│   ├── models/
│   │   └── face_resnet_state.pt
│   ├── outputs/
│   │   └── predictions.csv
│   └── src/
│       ├── face_detection.py
│       ├── model.py
│       ├── train_resnet.py
│       ├── predict_resnet.py
│       └── demo_pipeline.py
├── In-sensor Computing Experiments/
│   ├── data/
│   │   ├── patterns.csv
│   │   └── zju_10db_dataset.npz
│   ├── models/
│   │   └── legacy_mlp_weights.npz
│   ├── outputs/
│   │   ├── convolution_demo.csv
│   │   ├── zju_predictions.csv
│   │   └── zju_summary.json
│   └── src/
│       ├── convolution_demo.py
│       ├── evaluate_mlp.py
│       ├── generate_dataset.py
│       └── zju_utils.py
├── requirements.txt
└── README.md
```

## Installation

Python 3.10 or later is recommended.

```bash
pip install -r requirements.txt
```

For Apple Silicon, CUDA, or Raspberry Pi deployments, install the PyTorch build
that matches the target platform.

## Face Recognition Simulations

This folder contains the cleaned face-detection and face-recognition workflow.
The original human-face images are not included. To reproduce the workflow, place
authorized images in the following class-folder layout:

```text
Face Recognition Simulations/data/train_cropped/0/*.png
Face Recognition Simulations/data/train_cropped/1/*.png
...
Face Recognition Simulations/data/test_cropped/0/*.png
Face Recognition Simulations/data/test_cropped/1/*.png
...
```

### Face Detection

The detection preprocessing follows the original experiment pipeline:

1. Convert the input image to grayscale.
2. Downsample the image by 3x.
3. Downsample again by 3x for Haar face detection.
4. Map the detected face coordinates back to the first downsampled image.
5. Crop a fixed 128 x 128 face region from the detected face location.

Run detection after placing authorized raw images in
`Face Recognition Simulations/data/raw_test/`:

```bash
cd "Face Recognition Simulations"
python src/face_detection.py
```

Detection overlays and cropped faces are written under:

```text
Face Recognition Simulations/outputs/
```

### Face Recognition

Evaluate the pretrained ResNet after placing authorized cropped test images in
`Face Recognition Simulations/data/test_cropped/`:

```bash
cd "Face Recognition Simulations"
python src/predict_resnet.py
```

The prediction table is saved to:

```text
Face Recognition Simulations/outputs/predictions.csv
```

### Single-Image Demo

Run the full example pipeline for one authorized raw test image:

```bash
cd "Face Recognition Simulations"
python src/demo_pipeline.py --image data/raw_test/example.png
```

The demo saves grayscale, detection, crop, and first-convolution feature-map
outputs under `Face Recognition Simulations/outputs/demo/`.

### Training

Retrain the ResNet after placing authorized cropped training images in
`Face Recognition Simulations/data/train_cropped/`:

```bash
cd "Face Recognition Simulations"
python src/train_resnet.py --epochs 50
```

The new state dict is saved to:

```text
Face Recognition Simulations/models/face_resnet_state.pt
```

## In-sensor Computing Experiments

This folder contains the cleaned 4 x 4 GCCD-array experiment for recognizing
the letters `Z`, `J`, and `U`.

The legacy workflow was:

1. Define ideal `4 x 4` binary patterns for `Z`, `J`, and `U`.
2. Generate noisy samples at 10 dB SNR.
3. Flatten each `4 x 4` pattern into 16 inputs.
4. Use a `16 -> 8 -> 3` MLP classifier.
5. Map the first weighted-sum layer to the GCCD array.

Run the cleaned evaluation:

```bash
cd "In-sensor Computing Experiments"
python src/evaluate_mlp.py
python src/convolution_demo.py
```

Expected MLP test accuracy from the included dataset and legacy weights:

```text
accuracy_float_first_layer: 0.9933333333333333
accuracy_rounded_first_layer: 0.9955555555555555
```

`src/convolution_demo.py` demonstrates the 4 x 4 convolution-style weighted-sum
operation and writes the result to:

```text
In-sensor Computing Experiments/outputs/convolution_demo.csv
```

## Notes for Public Release

- Human-face image files are excluded.
- Generated caches such as `__pycache__`, `.DS_Store`, and temporary detection
  crops are ignored.
- All project explanations are kept in this single root `README.md`.
