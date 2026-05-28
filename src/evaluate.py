"""
Evaluation utilities.

Key entry points:
  evaluate()           — full metrics on a DataLoader
  breakdown_by_transform() — real/fake acc split by transformation type
  plot_confusion_matrix()
  plot_ablation()
"""

from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import REAL_FAKE_NAMES, TRANSFORM_NAMES
from .model import MultiTaskDetector


@torch.no_grad()
def evaluate(
    model: MultiTaskDetector,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, object]:
    """
    Returns overall accuracy + per-sample predictions for further analysis.
    """
    model.eval()
    all_rf_pred, all_rf_true = [], []
    all_tr_pred, all_tr_true = [], []

    for imgs, rf_labels, tr_labels in tqdm(loader, desc="  eval ", leave=False):
        imgs = imgs.to(device)
        rf_logits, tr_logits = model(imgs)
        all_rf_pred.extend(rf_logits.argmax(1).cpu().tolist())
        all_rf_true.extend(rf_labels.tolist())
        all_tr_pred.extend(tr_logits.argmax(1).cpu().tolist())
        all_tr_true.extend(tr_labels.tolist())

    rf_pred = np.array(all_rf_pred)
    rf_true = np.array(all_rf_true)
    tr_pred = np.array(all_tr_pred)
    tr_true = np.array(all_tr_true)

    return {
        "rf_acc": (rf_pred == rf_true).mean(),
        "tr_acc": (tr_pred == tr_true).mean(),
        "rf_pred": rf_pred,
        "rf_true": rf_true,
        "tr_pred": tr_pred,
        "tr_true": tr_true,
    }


def breakdown_by_transform(results: dict) -> dict[str, dict[str, float]]:
    """
    Break real/fake accuracy down by transformation class and by real/fake class.

    Returns dict: transform_name -> {overall, real, ai} accuracy.
    """
    rf_pred = results["rf_pred"]
    rf_true = results["rf_true"]
    tr_true = results["tr_true"]

    breakdown = {}
    for tr_label, tr_name in TRANSFORM_NAMES.items():
        mask = tr_true == tr_label
        if mask.sum() == 0:
            continue
        preds = rf_pred[mask]
        labels = rf_true[mask]
        overall_acc = (preds == labels).mean()

        sub = {}
        for rf_label, rf_name in REAL_FAKE_NAMES.items():
            class_mask = labels == rf_label
            if class_mask.sum() == 0:
                sub[rf_name] = float("nan")
            else:
                sub[rf_name] = (preds[class_mask] == labels[class_mask]).mean()

        breakdown[tr_name] = {"overall": overall_acc, **sub}

    return breakdown


def print_breakdown(breakdown: dict) -> None:
    header = f"{'Transform':<12} {'Overall':>8}  {'real':>8}  {'ai':>8}"
    print(header)
    print("-" * len(header))
    for tr_name, accs in breakdown.items():
        print(
            f"{tr_name:<12} {accs['overall']:>8.3f}  "
            f"{accs.get('real', float('nan')):>8.3f}  "
            f"{accs.get('ai', float('nan')):>8.3f}"
        )


def _confusion_matrix(true: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(true, pred):
        cm[t][p] += 1
    return cm


def plot_confusion_matrix(
    results: dict,
    save_path: Path | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    rf_cm = _confusion_matrix(results["rf_true"], results["rf_pred"], 2)
    rf_labels = list(REAL_FAKE_NAMES.values())
    sns.heatmap(
        rf_cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=rf_labels, yticklabels=rf_labels, ax=axes[0],
    )
    axes[0].set_title("Real / Fake")
    axes[0].set_ylabel("True")
    axes[0].set_xlabel("Predicted")

    tr_cm = _confusion_matrix(results["tr_true"], results["tr_pred"], 3)
    tr_labels = list(TRANSFORM_NAMES.values())
    sns.heatmap(
        tr_cm, annot=True, fmt="d", cmap="Greens",
        xticklabels=tr_labels, yticklabels=tr_labels, ax=axes[1],
    )
    axes[1].set_title("Transformation Type")
    axes[1].set_ylabel("True")
    axes[1].set_xlabel("Predicted")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_ablation(
    alphas: list[float],
    rf_accs: list[float],
    tr_accs: list[float],
    save_path: Path | None = None,
) -> None:
    """Plot real/fake acc and transform acc vs alpha (task weight)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(alphas, rf_accs, "o-", label="Real/Fake acc", color="steelblue")
    ax.plot(alphas, tr_accs, "s--", label="Transform acc", color="darkorange")
    ax.set_xlabel("α  (weight of real/fake loss)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Ablation: task loss weighting")
    ax.legend()
    ax.set_xticks(alphas)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_breakdown_heatmap(
    breakdown: dict[str, dict[str, float]],
    save_path: Path | None = None,
) -> None:
    """Heatmap of real/fake accuracy per transformation × class."""
    transforms = list(breakdown.keys())
    classes = ["real", "ai"]
    matrix = np.array([[breakdown[t].get(c, float("nan")) for c in classes] for t in transforms])

    fig, ax = plt.subplots(figsize=(5, 3))
    sns.heatmap(
        matrix, annot=True, fmt=".3f", vmin=0.0, vmax=1.0, cmap="RdYlGn",
        xticklabels=classes, yticklabels=transforms, ax=ax,
    )
    ax.set_title("Real/Fake accuracy by transformation & class")
    ax.set_xlabel("True class")
    ax.set_ylabel("Transformation")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_history(
    history: list[dict],
    save_path: Path | None = None,
) -> None:
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(epochs, [r["train_loss"] for r in history], label="train")
    axes[0].plot(epochs, [r["val_loss"] for r in history], label="val")
    axes[0].set_title("Total loss")
    axes[0].legend()

    axes[1].plot(epochs, [r["train_rf_acc"] for r in history], label="train")
    axes[1].plot(epochs, [r["val_rf_acc"] for r in history], label="val")
    axes[1].set_title("Real/Fake accuracy")
    axes[1].legend()

    axes[2].plot(epochs, [r["train_tr_acc"] for r in history], label="train")
    axes[2].plot(epochs, [r["val_tr_acc"] for r in history], label="val")
    axes[2].set_title("Transform accuracy")
    axes[2].legend()

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
