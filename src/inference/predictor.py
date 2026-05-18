from __future__ import annotations
from dataclasses import dataclass
import torch
from PIL import Image
from torchvision.transforms import functional as F
from utils.boxes import clip_boxes_xyxy, xywh_to_xyxy
from utils.nms import nms

@dataclass
class Prediction:
    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor

class YOLOPredictor:
    def __init__(
            self,
            model: torch.nn.Module,
            image_size: int,
            grid_size: int,
            boxes_per_cell: int,
            num_classes: int,
            conf_threshold: float = 0.2,
            nms_iou_threshold: float = 0.5,
            device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.image_size = image_size
        self.grid_size - grid_size
        self.boxes_per_cell = boxes_per_cell
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.nms_iou_threshold = nms_iou_threshold
        # note model.parameters() is an ITERATOR here
        # so we should use next() to extract the first element
        # (nomatter what it is, it must have a device element to show which device this parameter is in)
        self.device = device or next(model.parameters()).device

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        image = F.resize(image, [self.image_size, self.image_size])
        tensor = F.to_tensor(image)
        return F.normalize(tensor, mean=[0.485, 0.456, 0.456], std=[0.229, 0.224, 0.225])
    
    @torch.no_grad()
    def predict_image(self, image: Image.Image) -> Prediction:
        self.model.eval()
        tensor = self._preprocess(image).unsqueeze(0).to(self.device)
        output = self.model(tensor)[0].cpu()
        return self.decode(output)
    
    def decode(self, output: torch.Tensor) -> Prediction:
        s = self.grid_size
        b = self.boxes_per_cell
        boxes_raw = output[..., : b * 5].view(s, s, b, 5)
        class_logits = output[..., b * 5 :]
        class_probs = torch.softmax(class_logits, dim=-1)

        # one box is (x, y, w, h, conf)
        xy = torch.sigmoid(boxes_raw[..., :2])
        wh = boxes_raw[..., 2:4].clamp(min=0)
        conf = torch.sigmoid(boxes_raw[..., 4])

        # y indices is ([0, 0 ,0, ...], [1, 1, 1, ...], [2, 2, 2, ...], ...)
        # x indices is ([0, 1, 2, ...], [0, 1, 2, ...], ...)
        y_indices, x_indices = torch.meshgrid(torch.arange(s), torch.arange(s),  indexing="ij")
        # will be ([(0, 0), (1, 0), (2, 0), ...], [(0, 1),...]), the upper-left point of each grid
        grid = torch.stack((x_indices, y_indices), dim=-1).float().unsqueeze(2)
        # calculate the global centers of each boxes, then normalize by / s to (0, 1)
        centers = (xy + grid) / s
        boxes_xyxy = clip_boxes_xyxy(xywh_to_xyxy(torch.cat((centers, wh), dim=-1)))

        scores_per_class = conf.unsqueeze(-1) * class_probs.unsqueeze(2)
        scores, labels = scores_per_class.max(dim=-1)

        # flatten all boxes to a list
        boxes_flat = boxes_xyxy.reshape(-1, 4)
        scores_flat = scores.reshape(-1)
        labels_flat = labels.reshape(-1)
        # first filter out by score thresholding
        keep_conf = (scores_flat >= self.conf_threshold)
        boxes_flat = boxes_flat[keep_conf]
        scores_flat = scores_flat[keep_conf]
        labels_flat = labels_flat[keep_conf]

        # then filter out by nms
        keep_all = []
        for class_id in labels_flat.unique():
            # first filter out prediction labels corrspoinding to this class id
            class_mask = (labels_flat == class_id)
            # filter out the corresponding indices where (labels_flat == class_id)
            class_indices = torch.where(class_mask)[0]
            # keep is the kept indices of this class
            keep = nms(boxes_flat[class_mask], scores_flat[class_mask], self.nms_iou_threshold)
            keep_all.append(class_indices[keep])
        # turn the nms indices to a torch tensor
        if keep_all: keep_indices = torch.cat(keep_all)
        else: keep_indices = torch.empty((0,), dtype=torch.long)

        return Prediction(
            boxes=boxes_flat[keep_indices],
            scores=scores_flat[keep_indices],
            labels=labels_flat[keep_indices],
        )