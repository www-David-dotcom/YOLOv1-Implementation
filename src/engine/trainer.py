from __future__ import annotations
import csv
from pathlib import Path

import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from engine.evaluator import evaluate_model
from utils.checkpoint import save_checkpoint

def train_one_epoch(
        model: torch.nn.Module,
        loader: DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for images, targets, _ in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(images)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * images.shape[0]
    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate_one_epoch(
        model: torch.nn.Module,
        loader: DataLoader,
        val_loss_fn: torch.nn.Module,
        device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    for images, targets, _ in tqdm(loader, desc="val", leave=False):
        images = images.to(device)
        targets = targets.to(device)
        predictions = model(images)
        loss = val_loss_fn(predictions, targets)
        total_loss += float(loss.item()) * images.shape[0]
    return total_loss / len(loader.dataset)


def train_model(
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epochs: int,
        checkpoint_dir: str | Path,
        log_path: str | Path,
        eval_config: dict,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    best_map = -1.0
    last_complete_epoch = 0

    def _handle_interrupt():
        print(
            f"\nInterrupted at epoch {epoch}. "
            f"Last saved checkpoint is from epoch {last_complete_epoch}.",
            file=sys.stderr,
        )
        file.flush()
        sys.exit(1)

    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "val_loss", "map_50", "map_70", "map_90", "lr"])
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            try:
                train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
                val_loss = validate_one_epoch(model, val_loader, loss_fn, device)
                metrics = evaluate_model(model, val_loader, device, eval_config)

                current_lr = optimizer.param_groups[0]["lr"]
                row = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "map_50": metrics.get("map_50", 0.0),
                    "map_70": metrics.get("map_70", 0.0),
                    "map_90": metrics.get("map_90", 0.0),
                    "lr": current_lr,
                }
                writer.writerow(row)
                file.flush()

                if scheduler is not None:
                    scheduler.step()

                save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, epoch, row)
                last_complete_epoch = epoch
                if row["map_50"] > best_map:
                    best_map = row["map_50"]
                    save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch, row)

                print(
                    f"epoch={epoch} loss={train_loss:.4f} val_loss={val_loss:.4f} "
                    f"mAP@0.5={row['map_50']:.5f} "
                    f"mAP@0.7={row['map_70']:.4f} "
                    f"mAP@0.9={row['map_90']:.4f} "
                    f"lr={current_lr:.2e}"
                )
            except KeyboardInterrupt:
                _handle_interrupt()
