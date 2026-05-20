from __future__ import annotations
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset
from datasets.transforms import YOLOTransform

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def detection_collate(batch):
    images, targets, raw_boxes = zip(*batch, strict=True)
    return torch.stack(list(images)), torch.stack(list(targets)), list(raw_boxes)

class FaceDetectionDataset(Dataset):
    def __init__(
            self,
            image_dir: str | Path,
            label_dir: str | Path,
            image_size: int,
            grid_size: int,
            boxes_per_cell: int,
            num_classes: int,
            train: bool = True,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.grid_size = grid_size
        self.boxes_per_cell = boxes_per_cell
        self.num_classes = num_classes
        self.transform = YOLOTransform(image_size=image_size, train=train)
        # rglob is recursive matching in directory hierachy of paths
        self.image_paths = sorted(
            path for path in self.image_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _read_labels(self, label_path: Path) -> torch.Tensor:
        if not label_path.exists():
            return torch.zeros((0, 5), dtype=torch.float32)

        rows: list[list[float]] = []
        with label_path.open("r", encoding="utf-8") as file:
            for line in file:
                stripped = line.strip()
                if not stripped:
                    continue
                values = [float(value) for value in stripped.split()]
                # one values line is (class, x, y, w, h)
                if len(values) != 5:
                    raise ValueError(f"Invalid label line in {label_path}: {line}")
                rows.append(values)
        if not rows:
            return torch.zeros((0, 5), dtype=torch.float32)
        return torch.tensor(rows, dtype=torch.float32)

    def encode_target(self, boxes: torch.Tensor) -> torch.Tensor:
        s = self.grid_size
        b = self.boxes_per_cell
        c = self.num_classes
        target = torch.zeros((s, s, b * 5 + c), dtype=torch.float32)

        for row in boxes:
            class_id, x, y, w, h = row.tolist()
            x = min(max(x, 0.0), 0.999999)
            y = min(max(y, 0.0), 0.999999)
            cell_x = int(x * s)
            cell_y = int(y * s)
            local_x = x * s - cell_x
            local_y = y * s - cell_y

            # find the first empty box slot in this cell
            filled = False
            for box_index in range(b):
                offset = box_index * 5
                if target[cell_y, cell_x, offset + 4] == 0:
                    target[cell_y, cell_x, offset:offset+5] = torch.tensor(
                        [local_x, local_y, w, h, 1.0], dtype=torch.float32
                    )
                    filled = True
                    break
            if not filled:
                # all B slots occupied, skip this box
                continue

            class_offset = b * 5 + int(class_id)
            target[cell_y, cell_x, class_offset] = 1.0

        return target

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]
        label_path = self.label_dir / image_path.relative_to(self.image_dir).with_suffix(".txt")

        image = Image.open(image_path)
        boxes = self._read_labels(label_path)
        image_tensor, boxes = self.transform(image, boxes)
        target = self.encode_target(boxes)
        return image_tensor, target, boxes
