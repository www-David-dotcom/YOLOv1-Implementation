from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from inference.predictor import YOLOPredictor
from utils.boxes import xywh_to_xyxy
from utils.metrics import precision_recall_ap

@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    eval_config: dict,
) -> dict[str, float]:
    dataset = loader.dataset
    predictor = YOLOPredictor(
        model=model,
        image_size=dataset.transform.image_size,
        grid_size=dataset.grid_size,
        boxes_per_cell=dataset.boxes_per_cell,
        num_classes=dataset.num_classes,
        conf_threshold=eval_config.get("conf_threshold", 0.2),
        nms_iou_threshold=eval_config.get("nms_iou_threshold", 0.5),
        device=device,
    )

    pred_boxes: list[torch.Tensor] = []
    pred_scores: list[torch.Tensor] = []
    target_boxes: list[torch.Tensor] = []

    model.eval()
    for images, _, raw_targets in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        outputs = model(images).cpu()

        for output, raw in zip(outputs, raw_targets, strict=True):
            prediction = predictor.decode(output)
            pred_boxes.append(prediction.boxes)
            pred_scores.append(prediction.scores)
            if raw.numel() == 0: target_boxes.append(torch.zeros((0, 4), dtype=torch.float32))
            else: target_boxes.append(xywh_to_xyxy(raw[:, 1:5]))

    result: dict[str, float] = {}
    thresholds = eval_config.get("map_iou_threshold", [0.5, 0.7, 0.9])
    for threshold in thresholds:
        # calculate precision, recall, and mAP@threhold
        pr = precision_recall_ap(pred_boxes, pred_scores, target_boxes, float(threshold))
        key = str(int(round(float(threshold) * 100)))
        result[f"map_{key}"] = float(pr["ap"])
    
    # calculate mAP@50-95
    coco_thresholds =[round(0.5 + 0.05 * index, 2) for index in range(10)]
    coco_aps = [float(precision_recall_ap(pred_boxes, pred_scores, target_boxes,threshold)["ap"] for threshold in coco_thresholds)]
    result["map_50_95"] = sum(coco_aps) / len(coco_aps)
    return result

