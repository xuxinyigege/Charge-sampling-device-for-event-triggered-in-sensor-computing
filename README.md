# Charge sampling device for event-triggered in-sensor computing

This repository contains code for the face recognition simulations used in the GCCD in-sensor-computing study.

**Face Recognition Simulations** includes Haar-based face detection and ResNet-based face recognition.

## Structure

```text
.
├── Face Recognition Simulations/
│   ├── data/
│   │   ├── raw_test/        
│   │   ├── train_cropped/     
│   │   └── test_cropped/    
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
├── requirements.txt
└── README.md
```

## Face Recognition Simulations
### Face Detection

The detection preprocessing follows the original experiment pipeline:

1. Convert the input image to grayscale.
2. Downsample the image by 3x.
3. Downsample again by 3x for Haar face detection.
4. Map the detected face coordinates back to the first downsampled image.
5. Crop a fixed 128 x 128 face region from the detected face location.

Run detection after placing raw images in
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

Evaluate the pretrained ResNet after placing cropped test images in
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

Run the full example pipeline for one raw test image:

```bash
cd "Face Recognition Simulations"
python src/demo_pipeline.py --image data/raw_test/example.png
```

The demo saves grayscale, detection, crop, and first-convolution feature-map
outputs under `Face Recognition Simulations/outputs/demo/`.

### Training

Retrain the ResNet after placing cropped training images in
`Face Recognition Simulations/data/train_cropped/`:

```bash
cd "Face Recognition Simulations"
python src/train_resnet.py --epochs 50
```

The new state dict is saved to:

```text
Face Recognition Simulations/models/face_resnet_state.pt
```
