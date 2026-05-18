from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
matplotlib.use("Agg")


def test_package_exports_match_script_imports() -> None:
    from losses import YOLOv1Loss
    from models import YOLOv1

    assert YOLOv1.__name__ == "YOLOv1"
    assert YOLOv1Loss.__name__ == "YOLOv1Loss"


def test_scripts_are_importable_from_repository_root() -> None:
    script = (
        "import sys;"
        "sys.path.insert(0, 'scripts');"
        "import train, evaluate, infer, plot_curves, prepare_data"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_predictor_initializes_and_decodes_empty_prediction() -> None:
    from inference import YOLOPredictor

    model = torch.nn.Linear(1, 1)
    predictor = YOLOPredictor(
        model=model,
        image_size=64,
        grid_size=2,
        boxes_per_cell=1,
        num_classes=1,
        conf_threshold=0.99,
        nms_iou_threshold=0.5,
        device=torch.device("cpu"),
    )

    output = torch.zeros((2, 2, 6), dtype=torch.float32)
    prediction = predictor.decode(output)

    assert predictor.grid_size == 2
    assert prediction.boxes.shape == (0, 4)
    assert prediction.scores.shape == (0,)
    assert prediction.labels.shape == (0,)


def test_precision_recall_ap_uses_predictions_from_all_images() -> None:
    from utils.metrics import precision_recall_ap

    pred_boxes = [
        torch.tensor([[0.0, 0.0, 0.2, 0.2]], dtype=torch.float32),
        torch.tensor([[0.4, 0.4, 0.8, 0.8]], dtype=torch.float32),
    ]
    pred_scores = [
        torch.tensor([0.9], dtype=torch.float32),
        torch.tensor([0.8], dtype=torch.float32),
    ]
    target_boxes = [
        torch.tensor([[0.0, 0.0, 0.2, 0.2]], dtype=torch.float32),
        torch.tensor([[0.4, 0.4, 0.8, 0.8]], dtype=torch.float32),
    ]

    metrics = precision_recall_ap(pred_boxes, pred_scores, target_boxes, iou_threshold=0.5)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["ap"] == 1.0


def test_evaluator_uses_configured_thresholds_and_coco_average(monkeypatch) -> None:
    import engine.evaluator as evaluator

    observed_thresholds: list[float] = []

    def fake_precision_recall_ap(
        pred_boxes: list[torch.Tensor],
        pred_scores: list[torch.Tensor],
        target_boxes: list[torch.Tensor],
        threshold: float,
    ) -> dict[str, float]:
        observed_thresholds.append(threshold)
        return {"ap": threshold}

    monkeypatch.setattr(evaluator, "precision_recall_ap", fake_precision_recall_ap)

    class Transform:
        image_size = 32

    class Dataset(torch.utils.data.Dataset):
        transform = Transform()
        grid_size = 1
        boxes_per_cell = 1
        num_classes = 1

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            image = torch.zeros((3, 32, 32), dtype=torch.float32)
            target = torch.zeros((1, 1, 6), dtype=torch.float32)
            raw = torch.tensor([[0.0, 0.5, 0.5, 0.2, 0.2]], dtype=torch.float32)
            return image, target, raw

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.param = torch.nn.Parameter(torch.zeros(()))

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            batch_size = images.shape[0]
            return torch.zeros((batch_size, 1, 1, 6), dtype=torch.float32)

    loader = torch.utils.data.DataLoader(
        Dataset(),
        batch_size=1,
        collate_fn=lambda batch: (
            torch.stack([item[0] for item in batch]),
            torch.stack([item[1] for item in batch]),
            [item[2] for item in batch],
        ),
    )

    result = evaluator.evaluate_model(
        Model(),
        loader,
        torch.device("cpu"),
        {"map_iou_thresholds": [0.5, 0.7], "conf_threshold": 0.99},
    )

    assert result["map_50"] == 0.5
    assert result["map_70"] == 0.7
    assert result["map_50_95"] == sum(round(0.5 + 0.05 * i, 2) for i in range(10)) / 10
    assert observed_thresholds[:2] == [0.5, 0.7]
    assert observed_thresholds[2:] == [round(0.5 + 0.05 * i, 2) for i in range(10)]


def test_draw_boxes_converts_normalized_coordinates_to_pixels() -> None:
    from utils.visualization import draw_boxes

    image = np.zeros((10, 20, 3), dtype=np.uint8)
    output = draw_boxes(
        image,
        torch.tensor([[0.1, 0.2, 0.9, 0.8]], dtype=torch.float32),
        torch.tensor([0.75], dtype=torch.float32),
    )

    assert output.shape == image.shape
    assert output.sum() > 0


def test_plot_training_history_reads_map_50_column(tmp_path: Path) -> None:
    from utils.visualization import plot_training_history

    history = tmp_path / "history.csv"
    output = tmp_path / "curve.png"
    history.write_text(
        "epoch,train_loss,map_50,map_70,map_90\n"
        "1,3.0,0.25,0.1,0.0\n"
        "2,2.0,0.5,0.2,0.0\n",
        encoding="utf-8",
    )

    plot_training_history(history, output)

    assert output.exists()
    assert output.stat().st_size > 0


def test_face_dataset_discovers_jpg_images(tmp_path: Path) -> None:
    from PIL import Image

    from datasets.face_dataset import FaceDetectionDataset

    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()
    Image.new("RGB", (8, 8), color="white").save(image_dir / "sample.jpg")
    (label_dir / "sample.txt").write_text("", encoding="utf-8")

    dataset = FaceDetectionDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        image_size=8,
        grid_size=1,
        boxes_per_cell=1,
        num_classes=1,
        train=False,
    )

    assert len(dataset) == 1


def test_all_source_modules_import() -> None:
    modules = [
        "datasets.converters",
        "datasets.face_dataset",
        "datasets.transforms",
        "engine.evaluator",
        "engine.trainer",
        "inference",
        "inference.predictor",
        "losses",
        "losses.yolo_v1_loss",
        "models",
        "models.backbone",
        "models.head",
        "models.yolo_v1",
        "utils.boxes",
        "utils.checkpoint",
        "utils.config",
        "utils.metrics",
        "utils.nms",
        "utils.visualization",
    ]

    for module in modules:
        importlib.import_module(module)
