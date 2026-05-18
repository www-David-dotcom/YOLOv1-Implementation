from __future__ import annotations
import csv
from pathlib import Path

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
        # clear the old acuumulated gradients in the last epoch
        optimizer.zero_grad(set_to_none=True) # set the grad to None instead of zero to save memory
        predictions = model(images)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()
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
) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    best_map = -1.0

    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "map_50", "map_70", "map_90"])
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
            metrics = evaluate_model(model, val_loader, device, eval_config)

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "map_50": metrics.get("map_50", 0.0),
                "map_70": metrics.get("map_70", 0.0),
                "map_90": metrics.get("map_90", 0.0)
            }
            writer.writerow(row)
            file.flush()

            save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, epoch, row)
            if row["map_50"] > best_map: 
                best_map = row["map_50"]
                save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch, row)
            
            print(
                f"epoch={epoch} loss={train_loss: .4f} "
                f"mAP@0.5={row["map_50"]:.5f}, maAP@0.7={row["map_70"]:.4f}, mAP@0.9={row["map_90"]:.4f}"
            )