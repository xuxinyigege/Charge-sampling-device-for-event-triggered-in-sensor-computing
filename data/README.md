# Data

Human-face image files are intentionally excluded from this repository.

Use the following layout with your own authorized images:

```text
raw_test/              # raw images for Haar detection demos
train_cropped/0/*.png  # 128 x 128 grayscale training crops
train_cropped/1/*.png
...
test_cropped/0/*.png   # 128 x 128 grayscale evaluation crops
test_cropped/1/*.png
...
```
