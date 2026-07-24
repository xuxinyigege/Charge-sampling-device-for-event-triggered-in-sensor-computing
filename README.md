# Face Recognition and 4 x 4 GCCD Array Simulations

This repository contains cleaned code for two experiments used in “Charge
sampling device for event-triggered in-sensor computing”:

1. face detection and face recognition;
2. 4 x 4 GCCD-array simulations for `Z`, `J`, and `U` letter recognition.

Human-face image files are not included in this public repository for privacy
and consent reasons. The folder structure is kept so users can place their own
authorized images in the expected locations.

## Folder Structure

```text
face_experiment_github/
├── data/
│   ├── raw_test/          # Put authorized raw test images here
│   ├── train_cropped/     # Put 128 x 128 training face crops here
│   └── test_cropped/      # Put 128 x 128 evaluation face crops here
├── models/
│   ├── model_final.pt         # Legacy checkpoint from the original experiment
│   └── face_resnet_state.pt   # Portable state-dict checkpoint
├── outputs/
│   └── predictions.csv        # Prediction table produced by the cleaned code
└── src/
    ├── face_detection.py
    ├── model.py
    ├── train_resnet.py
    ├── predict_resnet.py
    ├── demo_pipeline.py
    └── export_legacy_state_dict.py
└── neuromorphic_zju/
    ├── data/
    ├── models/
    ├── outputs/
    └── src/
```

## Installation

Python 3.10 or later is recommended.

```bash
pip install -r requirements.txt
```

For Apple Silicon, CUDA, or Raspberry Pi deployments, install the PyTorch build
that matches the target platform.

## Face Detection

The detection preprocessing follows the experiment pipeline:

1. Convert the original image to grayscale.
2. Downsample by 3x.
3. Downsample again by 3x for Haar face detection.
4. Map the detected coordinates back to the first downsampled image.
5. Crop a fixed 128 x 128 face region from the top-left corner of the detected
   face box.

Run detection after placing authorized raw images in `data/raw_test/`:

```bash
python src/face_detection.py
```

Outputs are written to:

```text
outputs/detected_crops/
outputs/detection_overlays/
```

## Face Recognition

Evaluate the pretrained ResNet after placing authorized cropped test images in
`data/test_cropped/`:

```bash
python src/predict_resnet.py
```

The prediction table is saved to:

```text
outputs/predictions.csv
```

## Single-Image Demo

Run the full example pipeline for one raw test image:

```bash
python src/demo_pipeline.py --image data/raw_test/example.png
```

This saves:

```text
outputs/demo/gray_downsampled.png
outputs/demo/face_detection.png
outputs/demo/face_crop.png
outputs/demo/feature_map_0.png
outputs/demo/feature_map_1.png
outputs/demo/feature_map_2.png
outputs/demo/feature_map_3.png
```

## Training

Retrain the ResNet after placing authorized cropped training images in
`data/train_cropped/`:

```bash
python src/train_resnet.py --epochs 50
```

The new state dict is saved to:

```text
models/face_resnet_state.pt
```

## Data Note

The original experiment used human-face images. Those files are intentionally
excluded from this repository. To reproduce the workflow, prepare your own
authorized images with the same class-folder layout:

```text
data/train_cropped/0/*.png
data/train_cropped/1/*.png
...
data/test_cropped/0/*.png
data/test_cropped/1/*.png
...
```

## Neuromorphic Z/J/U Letter Recognition

The `neuromorphic_zju/` folder contains the cleaned 4 x 4 GCCD-array experiment
for recognizing the letters `Z`, `J`, and `U`.

The legacy MATLAB workflow was:

1. define ideal `4 x 4` binary patterns for `Z`, `J`, and `U`;
2. generate noisy samples at 10 dB SNR;
3. flatten each `4 x 4` pattern into 16 inputs;
4. use a `16 -> 8 -> 3` MLP classifier;
5. map the first weighted-sum layer to the GCCD array.

Run the cleaned evaluation:

```bash
cd neuromorphic_zju
python src/evaluate_mlp.py
python src/convolution_demo.py
```

Expected MLP test accuracy:

```text
accuracy_float_first_layer: 0.9933333333333333
accuracy_rounded_first_layer: 0.9955555555555555
```

See `neuromorphic_zju/README.md` for details.
