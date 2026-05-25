# YOLOv1 Face Detection on WIDER FACE

## 1. Introduction

This project implements a YOLO-style object detector from scratch in PyTorch and trains it for face detection on the WIDER FACE dataset. The file structure, and instructions to train, validate and visualise the model are included in `README.md`.

## 2. Dataset and Preprocessing

The original WIDER FACE annotations were converted into YOLO-format text labels:

```text
class_id x_center y_center width height
```

All box coordinates are normalized to `[0, 1]`. The processed dataset used 12800 training pictures and 3226 validation pictures.

Images are resized to `448 x 448`. During training, horizontal flip, color jitter, blur, cutout, and small affine transforms were tested as regularization. Validation preprocessing is deterministic.

## 3. Model Architecture

The detector follows a compact YOLOv1-style design according to the thesis:

- A convolutional backbone made from `Conv2d + BatchNorm2d + LeakyReLU` blocks.
- A detection head with several convolutional layers and dropout. The initial MLP head proposed in the paper not only has much more parameters but also performed worse in early experiments, so the convolutional head was used for all experiments.
- Output shape: `S x S x (B * 5 + C)`, where $C = 1$ because there is only one class (face). 

The main training configuration:

| Parameter | Value |
|---|---:|
| Image size | 448 |
| Grid size | 28 |
| Boxes per cell | 2 |
| Batch size | 8 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 5e-4 to 1e-3 |
| Scheduler | Cosine annealing |
| Dropout | 0.5 |

## 4. Loss and Inference

According to the paper, the YOLO loss contains:

- Coordinate loss for box center and size.
- Objectness loss for cells responsible for faces.
- No-object confidence loss for background cells.
- Classification loss for the face class.

and how it combines them is like 

```text
loss = lambda_coord * coord_loss
     + object_loss
     + lambda_noobj * noobject_loss
     + class_loss
```

For inference, predictions are decoded from cell-relative coordinates into normalized image coordinates. Low-confidence boxes are filtered, and NMS is applied to remove duplicate detections.

## 5. Experimental Results
I trained on NVIDIA A100 40GB GPU, where each training epoch of 1610 images takes about 5 minutes, and evaluation on 404 validation images per epoch takes about 1.5 minutes. The training was run for 100 epochs, and the best model was selected based on validation mAP@0.5.

We can see that both training and validation loss decreased, but after 50 epochs, validation loss plateaued near `6.30`, and training loss continued decreasing, and `mAP@0.5` plateaued around `0.30`. This is a typical overfitting, so the best model was selected based on the best validation mAP@0.5, which occurred around epoch 50. The final evaluation metrics were computed on this best model.

### Training and Validation Loss

![Reconstructed loss curve](outputs/report/reconstructed_loss_curve.png)

The loss curve shows stable convergence. The validation loss initially decreases quickly and then reaches a plateau around `6.30`. After this point, the training loss continues decreasing, which indicates overfitting to the training set.

### mAP Curves

![Reconstructed mAP curve](outputs/report/reconstructed_map_curve.png)

The best observed performance was approximately:

| Metric | Approximate value |
|---|---:|
| mAP@0.5 | 0.30 |
| mAP@0.7 | 0.09 to 0.10 |
| mAP@0.9 | close to 0 |

The gap between `mAP@0.5` and `mAP@0.9` suggests that the detector often finds approximate face regions but does not localize boxes tightly enough for high-IoU evaluation.

### Precision-Recall Curve at IoU 0.7

![Reconstructed PR curve](outputs/report/reconstructed_pr_iou70.png)

At IoU `0.7`, precision drops quickly as recall increases. This is consistent with the observed low `mAP@0.7`: many predictions are either loose, duplicated, or false positives.

## 6. Qualitative Result

An example inference output is shown below.

![Inference example](outputs/predictions/infer.jpg)

We can see that the model detects many faces, but most boxes are clustered together. Due to this, I reduced the NMS IoU threshold from `0.5` to `0.3` to reduce duplicates, which makes mAP@0.5 much higher in the early training stages.

## 7. Hyperparameter Tuning and Discussion

Several tuning experiments were performed during development.

### Grid Size

The grid size was increased from `14 x 14` to `28 x 28`. This was important because WIDER FACE contains many small faces. With `448 x 448` input:

- `14 x 14` gives one cell per `32 x 32` pixels.
- `28 x 28` gives one cell per `16 x 16` pixels.

Since many WIDER FACE boxes are smaller than `16` to `32` pixels, the finer grid improved the model's ability to represent crowded small faces.

### Boxes Per Cell

Experiments used up to `5` boxes per cell. Increasing `B` improves capacity in crowded scenes, but it also increases the number of background predictions. This caused slower evaluation and more false positives. So I fell back to `2` boxes per cell at last, because it performs better in terms of mAP.


### Overfitting

The model began to overfit after validation loss reached its plateau. The following regularization methods were added after this has been spotted:

- Dropout in the backbone/head.
- AdamW weight decay.
- Horizontal flip and photometric augmentation.
- Stronger augmentation such as blur, cutout, and affine transforms.

Stronger augmentation reduced overfitting very much. Without these augmentations (only horizontal flip), mAP@0.5 will even stay at a poor `1e-5` plateau.

### False Positives

Visual inspection showed that the model sometimes produced many high-confidence background boxes.
![False Positives](outputs/predictions/inspect_best/01_0_Parade_Parade_0_102.jpg)

 This suggests that the confidence/objectness head needs stronger background calibration. Increasing `lambda_noobj`, using a stricter confidence threshold, and reducing boxes per cell are practical ways to reduce this issue.

## 8. Conclusion
Because time is limited, I did not inplement any further versions of YOLO. Future improvements of YOLOv3 would include anchor-based prediction, focal loss or BCE objectness loss, better confidence calibration, stronger NMS variants, and a larger backbone pretrained on ImageNet.
