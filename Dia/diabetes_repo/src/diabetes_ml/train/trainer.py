from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    lr: float
    batch_size: int
    epochs: int
    patience: int
    weight_decay: float
    label_smoothing: float
    grad_clip: float
    amp: bool


def compute_class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes)
    inv = 1.0 / (counts + 1e-9)
    w = inv / inv.sum() * n_classes
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def predict_logits_proba(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_y = [], []
    for batch in loader:
        cat = batch["cat"].to(device, non_blocking=True)
        num = batch["num"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        logits = model(cat, num)
        all_logits.append(logits.detach().float().cpu())
        all_y.append(y.detach().cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    y = torch.cat(all_y, dim=0).numpy()
    proba = F.softmax(torch.tensor(logits), dim=1).numpy()
    return logits, y, proba


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.amp.GradScaler],
    criterion: Optional[nn.Module],
    use_amp: bool,
    grad_clip: float,
) -> float:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_n = 0.0, 0

    for batch in loader:
        cat = batch["cat"].to(device, non_blocking=True)
        num = batch["num"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True) if (is_train and use_amp and device.type == "cuda") else nullcontext()

        with amp_ctx:
            logits = model(cat, num)
            loss = criterion(logits, y) if criterion is not None else None

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if use_amp and device.type == "cuda" and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip and grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip and grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        bs = y.size(0)
        total_n += bs
        total_loss += float(loss.item()) * bs if loss is not None else 0.0

    return total_loss / max(1, total_n)


def train_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
    n_classes: int,
    y_train: np.ndarray,
    metric_fn,  # lower is better (e.g., brier)
    scheduler_factory=None,
):
    cls_w = compute_class_weights(y_train, n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=cfg.label_smoothing).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = scheduler_factory(optimizer) if scheduler_factory is not None else None

    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_metric = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        tr_loss = run_epoch(model, train_loader, device, optimizer, scaler, criterion, use_amp, cfg.grad_clip)

        _, y_val, proba_val = predict_logits_proba(model, val_loader, device)
        val_metric = float(metric_fn(y_val, proba_val))

        if scheduler is not None:
            scheduler.step()

        print(f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} | val_metric={val_metric:.4f}")

        if val_metric + 1e-6 < best_metric:
            best_metric = val_metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_metric
