"""
Entry point for all three project phases.

Usage:
  python main.py                          # run all phases with configs/config.yaml
  python main.py --config configs/config.yaml --phase baselines
  python main.py --phase joint
  python main.py --phase ablation
  python main.py --phase analysis --checkpoint runs/joint/best.pt
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import (
    RRDataset, build_samples, make_splits, print_split_stats
)
from src.model import MultiTaskDetector
from src.train import train
from src.evaluate import (
    evaluate, breakdown_by_transform, print_breakdown,
    plot_confusion_matrix, plot_ablation, plot_breakdown_heatmap,
    plot_training_history,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    print(f"Using device: {dev}")
    return dev


def make_loaders(cfg: dict, train_split, val_split, test_split):
    dc = cfg["data"]
    tc = cfg["training"]
    train_ds = RRDataset(train_split, image_size=dc["image_size"], augment=True)
    val_ds   = RRDataset(val_split,   image_size=dc["image_size"], augment=False)
    test_ds  = RRDataset(test_split,  image_size=dc["image_size"], augment=False)

    nw = tc.get("num_workers", 2)
    train_loader = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True,
                              num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=tc["batch_size"], shuffle=False,
                              num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=tc["batch_size"], shuffle=False,
                              num_workers=nw, pin_memory=True)
    return train_loader, val_loader, test_loader


def run_one(
    name: str,
    alpha: float,
    cfg: dict,
    train_loader, val_loader, test_loader,
    device: torch.device,
    plots_dir: Path,
) -> dict[str, float]:
    mc = cfg["model"]
    tc = cfg["training"]
    runs_dir = Path(cfg["output"]["runs_dir"]) / name

    print(f"\n{'='*60}")
    print(f"  Run: {name}  (alpha={alpha})")
    print(f"{'='*60}")

    model = MultiTaskDetector(
        dropout=mc["dropout"],
        freeze_backbone=mc["freeze_backbone"],
    ).to(device)

    history = train(
        model, train_loader, val_loader,
        epochs=tc["epochs"],
        lr=tc["lr"],
        weight_decay=tc["weight_decay"],
        alpha=alpha,
        device=device,
        run_dir=runs_dir,
        patience=tc["patience"],
    )

    plot_training_history(history, save_path=plots_dir / f"{name}_history.png")

    # Load best checkpoint for test evaluation
    ckpt = torch.load(runs_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    results = evaluate(model, test_loader, device)
    print(f"\n  Test  rf_acc={results['rf_acc']:.4f}  tr_acc={results['tr_acc']:.4f}")

    plot_confusion_matrix(results, save_path=plots_dir / f"{name}_confusion.png")

    breakdown = breakdown_by_transform(results)
    print("\n  Real/Fake accuracy by transformation:")
    print_breakdown(breakdown)
    plot_breakdown_heatmap(breakdown, save_path=plots_dir / f"{name}_breakdown.png")

    # Persist metrics
    metrics = {"rf_acc": float(results["rf_acc"]), "tr_acc": float(results["tr_acc"])}
    with open(runs_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ── Phases ────────────────────────────────────────────────────────────────────

def phase_baselines(cfg, train_loader, val_loader, test_loader, device, plots_dir):
    results = {}
    for spec in cfg["phases"]["baselines"]:
        m = run_one(spec["name"], spec["alpha"], cfg,
                    train_loader, val_loader, test_loader, device, plots_dir)
        results[spec["name"]] = m
    return results


def phase_joint(cfg, train_loader, val_loader, test_loader, device, plots_dir):
    spec = cfg["phases"]["joint"]
    m = run_one(spec["name"], spec["alpha"], cfg,
                train_loader, val_loader, test_loader, device, plots_dir)
    return {spec["name"]: m}


def phase_ablation(cfg, train_loader, val_loader, test_loader, device, plots_dir):
    alphas = cfg["phases"]["ablation"]["alphas"]
    rf_accs, tr_accs = [], []

    for alpha in alphas:
        name = f"ablation_a{str(alpha).replace('.', '')}"
        m = run_one(name, alpha, cfg,
                    train_loader, val_loader, test_loader, device, plots_dir)
        rf_accs.append(m["rf_acc"])
        tr_accs.append(m["tr_acc"])

    print("\n=== Ablation summary ===")
    for a, rf, tr in zip(alphas, rf_accs, tr_accs):
        print(f"  alpha={a:.1f}  rf_acc={rf:.4f}  tr_acc={tr:.4f}")

    plot_ablation(alphas, rf_accs, tr_accs, save_path=plots_dir / "ablation.png")
    return {"alphas": alphas, "rf_accs": rf_accs, "tr_accs": tr_accs}


def phase_analysis(cfg, test_loader, device, checkpoint: str):
    """Load a trained model and produce all analysis plots (no training)."""
    plots_dir = Path(cfg["output"]["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    mc = cfg["model"]
    model = MultiTaskDetector(dropout=mc["dropout"]).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint: {checkpoint}  (alpha={ckpt.get('alpha')})")

    results = evaluate(model, test_loader, device)
    print(f"rf_acc={results['rf_acc']:.4f}  tr_acc={results['tr_acc']:.4f}")

    plot_confusion_matrix(results, save_path=plots_dir / "analysis_confusion.png")
    breakdown = breakdown_by_transform(results)
    print_breakdown(breakdown)
    plot_breakdown_heatmap(breakdown, save_path=plots_dir / "analysis_breakdown.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--phase",
        choices=["all", "baselines", "joint", "ablation", "analysis"],
        default="all",
    )
    parser.add_argument("--checkpoint", default=None,
                        help="Path to .pt file (required for --phase analysis)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    plots_dir = Path(cfg["output"]["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── Data preparation ──────────────────────────────────────────────────────
    print("\nScanning dataset…")
    dc = cfg["data"]
    samples = build_samples(
        root=dc["root"],
        subset_per_cell=dc.get("subset_per_cell"),
        seed=dc["seed"],
    )
    train_split, val_split, test_split = make_splits(
        samples,
        train_frac=dc["train_frac"],
        val_frac=dc["val_frac"],
        seed=dc["seed"],
    )
    print_split_stats(train_split, val_split, test_split)
    train_loader, val_loader, test_loader = make_loaders(
        cfg, train_split, val_split, test_split
    )

    # ── Run requested phase(s) ────────────────────────────────────────────────
    if args.phase == "analysis":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for --phase analysis")
        phase_analysis(cfg, test_loader, device, args.checkpoint)
        return

    all_metrics = {}

    if args.phase in ("all", "baselines"):
        all_metrics.update(phase_baselines(cfg, train_loader, val_loader, test_loader, device, plots_dir))

    if args.phase in ("all", "joint"):
        all_metrics.update(phase_joint(cfg, train_loader, val_loader, test_loader, device, plots_dir))

    if args.phase in ("all", "ablation"):
        all_metrics["ablation"] = phase_ablation(
            cfg, train_loader, val_loader, test_loader, device, plots_dir
        )

    # Summary table
    print("\n\n=== Final test results across all runs ===")
    print(f"{'Run':<20} {'RF acc':>8} {'TR acc':>8}")
    print("-" * 40)
    for name, m in all_metrics.items():
        if isinstance(m, dict) and "rf_acc" in m:
            print(f"{name:<20} {m['rf_acc']:>8.4f} {m['tr_acc']:>8.4f}")

    with open(plots_dir / "all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\nAll plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()
