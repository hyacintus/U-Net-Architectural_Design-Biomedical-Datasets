# U-Net Medical Image Segmentation

## Overview

This repository contains three Python scripts that train U-Net models for semantic segmentation on medical image datasets. Each script automatically loads images, preprocesses them, splits the data into training, validation, and test sets, and evaluates the model on multiple metrics. Results are saved in a timestamped CSV file for easy comparison.

All scripts support **grid search** over U-Net architecture configurations: the number of encoder/decoder steps and the number of base filters can be set as lists, and the training loop will automatically run one model per combination.

---

## Repository Structure

```
├── unet_segmentation_brain.py   # U-Net for brain tumor segmentation (TIFF, patch-free)
├── unet_segmentation_breast.py  # U-Net for breast lesion segmentation (TIFF, sliding-window patches)
├── unet_segmentation_skin.py    # U-Net for skin lesion segmentation (JPG/PNG, patch-free)
└── README.md
```

---

## Dataset Download

Please download the following datasets from Kaggle and place them in a folder of your choice. Then update the `image_dir` and `mask_dir` paths at the top of the corresponding script.

| Script | Dataset | Kaggle Link |
|--------|---------|-------------|
| `unet_segmentation_brain.py` | Brain Tumor Dataset H | [Download](https://kaggle.com/datasets/4e5596da6b1ebfa9044ac3ddf2a424d12968f6ee193ec7d3b02aa31b7247d5df) |
| `unet_segmentation_breast.py` | CBIS-DDSM-H | [Download](https://kaggle.com/datasets/9cef8368a7037c7d63324c73fa6384c0e6f9d02e10c2b07a0c37b640dbf17c45) |
| `unet_segmentation_skin.py` | ISIC-2017-H | [Download](https://kaggle.com/datasets/4e83433e33e4b0eb6fa18f9ad6956a3340d317ac4156d9538ed95704a0db2b62) |

---

## Dataset Details

### Brain Tumor Dataset H — `unet_segmentation_brain.py`
MRI brain scans in TIFF format with binary segmentation masks (tumor vs. background). Images are 512×512 with 3 channels. The script loads up to `MAX_IMAGES` samples, normalizes pixel values, and splits the dataset into train/validation/test sets.

### CBIS-DDSM-H — `unet_segmentation_breast.py`
Mammography images in TIFF format (grayscale, 1024×1024) with binary lesion masks. Due to the large image size, this script uses a **sliding-window patch extraction** strategy: each image is divided into 128×128 patches with 50% overlap. CLAHE contrast enhancement is applied before training. Patches are filtered to retain those with sufficient lesion content.

### ISIC-2017-H — `unet_segmentation_skin.py`
Dermoscopy images in JPG format with PNG binary segmentation masks (skin lesion vs. background). Images are resized to 512×512 with 3 channels. The script loads up to `MAX_IMAGES` samples and normalizes pixel values to [0, 1].

---

## Configuration

### Path Configuration

At the top of each script, set the paths to your images and masks:

```python
image_dir = "path/to/images"
mask_dir  = "path/to/masks"
```

### Architecture Configuration

Each script supports grid search over two parameters:

```python
list_num_steps = [5]         # Number of encoder/decoder levels   e.g. [3, 4, 5]
list_base_filters = [64]     # Filters at the first level          e.g. [4, 8, 16, 32, 64]
```

The number of filters doubles at each level. One model is trained for every combination in the two lists. For example:

```python
list_num_steps    = [3, 5]
list_base_filters = [32, 64]
```

will train 4 models: (steps=3, filters=32), (steps=3, filters=64), (steps=5, filters=32), (steps=5, filters=64).

---

## Output

Each run produces a timestamped CSV file (e.g. `segmentation_metrics_Brain_28Jun2026_10h30m.csv`) containing one row per trained model, with the following metrics:

- Sensitivity, Specificity, Precision, Recall
- IoU, Mean IoU, Frequency-Weighted IoU (FW-IoU)
- Dice / F1 Score
- Pixel Accuracy, Mean Accuracy
- Training time (minutes)
- Per-epoch train/validation Dice and Loss

---

## Requirements

```
tensorflow >= 2.x
numpy
pandas
scikit-learn
scikit-image
tifffile          # brain and breast scripts
opencv-python     # breast script
tqdm
```

Install all dependencies with:

```bash
pip install tensorflow numpy pandas scikit-learn scikit-image tifffile opencv-python tqdm
```

---

## Usage

```bash
python unet_segmentation_brain.py
python unet_segmentation_breast.py
python unet_segmentation_skin.py
```
