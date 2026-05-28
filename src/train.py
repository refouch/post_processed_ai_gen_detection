"""Training loop for the multi-task detector."""

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import MultiTaskDetector, multitask_loss


def train_one_epoch(
    model: MultiTaskDetector,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    alpha: float,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = rf_loss_sum = tr_loss_sum = 0.0
    rf_correct = tr_correct = n = 0

    for imgs, rf_labels, tr_labels in tqdm(loader, desc="  train", leave=False):
        imgs = imgs.to(device)
        rf_labels = rf_labels.to(device)
        tr_labels = tr_labels.to(device)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.autocast(device_type=device.type):
                rf_logits, tr_logits = model(imgs)
                loss, rf_loss, tr_loss = multitask_loss(
                    rf_logits, tr_logits, rf_labels, tr_labels, alpha
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            rf_logits, tr_logits = model(imgs)
            loss, rf_loss, tr_loss = multitask_loss(
                rf_logits, tr_logits, rf_labels, tr_labels, alpha
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        bs = imgs.size(0)
        n += bs
        total_loss += loss.item() * bs
        rf_loss_sum += rf_loss.item() * bs
        tr_loss_sum += tr_loss.item() * bs
        rf_correct += (rf_logits.argmax(1) == rf_labels).sum().item()
        tr_correct += (tr_logits.argmax(1) == tr_labels).sum().item()

    return {
        "loss": total_loss / n,
        "rf_loss": rf_loss_sum / n,
        "tr_loss": tr_loss_sum / n,
        "rf_acc": rf_correct / n,
        "tr_acc": tr_correct / n,
    }


@torch.no_grad()
def validate(
    model: MultiTaskDetector,
    loader: DataLoader,
    alpha: float,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = rf_loss_sum = tr_loss_sum = 0.0
    rf_correct = tr_correct = n = 0

    for imgs, rf_labels, tr_labels in tqdm(loader, desc="  val  ", leave=False):
        imgs = imgs.to(device)
        rf_labels = rf_labels.to(device)
        tr_labels = tr_labels.to(device)

        rf_logits, tr_logits = model(imgs)
        loss, rf_loss, tr_loss = multitask_loss(
            rf_logits, tr_logits, rf_labels, tr_labels, alpha
        )

        bs = imgs.size(0)
        n += bs
        total_loss += loss.item() * bs
        rf_loss_sum += rf_loss.item() * bs
        tr_loss_sum += tr_loss.item() * bs
        rf_correct += (rf_logits.argmax(1) == rf_labels).sum().item()
        tr_correct += (tr_logits.argmax(1) == tr_labels).sum().item()

    return {
        "loss": total_loss / n,
        "rf_loss": rf_loss_sum / n,
        "tr_loss": tr_loss_sum / n,
        "rf_acc": rf_correct / n,
        "tr_acc": tr_correct / n,
    }


def train(
    model: MultiTaskDetector,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    alpha: float,
    device: torch.device,
    run_dir: Path,
    patience: int = 5,
) -> list[dict]:
    """
    Full training run. Saves best checkpoint (by val rf_acc when alpha > 0,
    else by val tr_acc). Returns history list for plotting.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    monitor_key = "rf_acc" if alpha > 0.0 else "tr_acc"
    best_val = 0.0
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, alpha, device, scaler)
        val_metrics = validate(model, val_loader, alpha, device)
        scheduler.step()

        elapsed = time.time() - t0
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
               **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)

        print(
            f"  Epoch {epoch:3d}/{epochs}"
            f"  loss={train_metrics['loss']:.4f}/{val_metrics['loss']:.4f}"
            f"  rf_acc={train_metrics['rf_acc']:.3f}/{val_metrics['rf_acc']:.3f}"
            f"  tr_acc={train_metrics['tr_acc']:.3f}/{val_metrics['tr_acc']:.3f}"
            f"  ({elapsed:.0f}s)"
        )

        if val_metrics[monitor_key] > best_val:
            best_val = val_metrics[monitor_key]
            no_improve = 0
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "alpha": alpha, "val_metrics": val_metrics},
                run_dir / "best.pt",
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    return history
