# YOLOv1 Face Detection

Train a small YOLOv1-style face detector on WIDER FACE.

## Setup

```bash
uv sync
```

Run all commands from the repository root.

## File Structure

```text
CV_Project3/
├── configs/
│   └── yolo_v1_face.yaml        # Dataset, model, training, and eval settings
├── scripts/
│   ├── download_widerface.py    # Download and extract WIDER FACE archives
│   ├── prepare_data.py          # Convert WIDER FACE annotations to YOLO labels
│   ├── train.py                 # Train the detector
│   ├── evaluate.py              # Evaluate checkpoints on validation data
│   ├── infer.py                 # Run inference on one image
│   └── plot_curves.py           # Plot training history
├── src/
│   ├── datasets/                # Dataset loader, transforms, annotation conversion
│   ├── engine/                  # Training and evaluation loops
│   ├── inference/               # Prediction decoding
│   ├── losses/                  # YOLO loss
│   ├── models/                  # Backbone, head, full YOLO model
│   └── utils/                   # Boxes, NMS, metrics, config, checkpoints
├── data/
│   ├── WIDER_train/             # Raw train images after extraction
│   ├── WIDER_val/               # Raw validation images after extraction
│   ├── WIDER_test/              # Raw test images after extraction
│   ├── wider_face_split/        # Original WIDER FACE annotation text files
│   └── processed/               # YOLO-format processed images and labels
├── outputs/
│   ├── checkpoints/             # Saved model checkpoints
│   ├── logs/                    # Training CSV logs
│   ├── predictions/             # Inference visualizations
│   └── report/                  # Report figures
├── README.md
├── report.md
└── pyproject.toml
```

## 1. Download WIDER FACE

Download the train, validation, test, and annotation archives from the Hugging Face dataset repo:

```bash
uv run python scripts/download_widerface.py --extract
```

This writes archives and extracted files under `data/`. The expected extracted layout is:

```text
data/WIDER_train/images/
data/WIDER_val/images/
data/WIDER_test/images/
data/wider_face_split/wider_face_train_bbx_gt.txt
data/wider_face_split/wider_face_val_bbx_gt.txt
```

Use `--force` to redownload existing archives. Use `HF_TOKEN` or `--token` if Hugging Face requires authentication.

## 2. Prepare YOLO Labels

Convert WIDER FACE train annotations to YOLO text labels:

```bash
uv run python scripts/prepare_data.py \
  --images-root data/WIDER_train/images \
  --annotation-file data/wider_face_split/wider_face_train_bbx_gt.txt \
  --output-images data/processed/images/train \
  --output-labels data/processed/labels/train \
  --skip-invalid
```

Convert validation annotations:

```bash
uv run python scripts/prepare_data.py \
  --images-root data/WIDER_val/images \
  --annotation-file data/wider_face_split/wider_face_val_bbx_gt.txt \
  --output-images data/processed/images/val \
  --output-labels data/processed/labels/val \
  --skip-invalid
```

`configs/yolo_v1_face.yaml` already points to these processed train/val folders. WIDER FACE test images do not include ground-truth labels, so they are mainly used for inference.

## 3. Train
Test if cuda is available:
```bash
uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```
train:
```bash
uv run python scripts/train.py --config configs/yolo_v1_face.yaml --gpu 1
```

Outputs:

```text
outputs/checkpoints/last.pt
outputs/checkpoints/best.pt
outputs/logs/train_history.csv
```

Plot the training curve:

```bash
uv run python scripts/plot_curves.py
```

## 4. Evaluate

Evaluate the best checkpoint on the validation split:

```bash
uv run python scripts/evaluate.py \
  --config configs/yolo_v1_face.yaml \
  --checkpoint outputs/checkpoints/best.pt
```

The evaluator reports mAP values for the IoU thresholds configured in `configs/yolo_v1_face.yaml`.

## 5. Run Inference

Run prediction on any image from train, validation, test, or your own image:

```bash
uv run python scripts/infer.py \
  --config configs/yolo_v1_face.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --image data/WIDER_test/images/0--Parade/0_Parade_marchingband_1_9.jpg \
  --output outputs/predictions/infer.jpg
```

For example, choose an image under `data/WIDER_test/images/` after extraction and pass that path as `--image`.

## Verification

```bash
uv run pytest -q
uv run ruff check .
```
