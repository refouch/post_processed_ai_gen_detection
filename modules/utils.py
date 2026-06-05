"""Useful functions, mainly visual stuff."""

import numpy as np
import matplotlib.pyplot as plt
import os


def show_sample(img_tensor, label, transform):
    """Code borrowed from tutorial n°2. Just added multi-label support"""

    img = img_tensor.numpy().transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    plt.imshow(img)
    plt.title(f"Class: {label}, Transform:{transform}, Shape: ({img_tensor.shape[0]},{img_tensor.shape[1]},{img_tensor.shape[2]})")
    plt.axis('off')
    plt.show()


def print_all_results(all_results, omit_breadown = False):
    """Pretty print in terminal for the result dictionnaries"""
    COL = 20
    W = 10

    # Summary table
    header = f"{'Model':<{COL}} {'Real/Fake':>{W}} {'Transform':>{W}}"
    sep = "─" * len(header)
    print(sep)
    print(header)
    print(sep)
    for name, res in all_results.items():
        rf = f"{res['acc_realfake']:.2%}" if 'acc_realfake' in res else "—"
        tf = f"{res['acc_transform']:.2%}" if 'acc_transform' in res else "—"
        print(f"{name:<{COL}} {rf:>{W}} {tf:>{W}}")
    print(sep)

    # Per-transform breakdowns 
    for name, res in all_results.items():
        sections = [
            ("rf_by_transform", "Real/Fake by transform"),
            ("tf_by_transform", "Transform id by transform"),
            ("rf_by_context",   "Real/Fake by context"),
        ]

        has_breakdown = any(res.get(k) for k, _ in sections)

        if not has_breakdown or omit_breadown: # If no breakdown OR omit then skip.
            continue

        print(f"\n  [{name}]")
        for key, label in sections:
            data = res.get(key)

            if not data:
                continue

            print(f"    {label}:")
            for k, v in sorted(data.items(), key=lambda x: str(x[0])):
                
                if type(k) == float: # Filter out residual nan
                    continue
                print(f"      {k:<22} {v:.2%}")


def plot_history(history, title="Training history", save_path=None):
    """Function to plot the training history. Produces 3 to 4 plots:
        - Training loss
        - Accuracy to the objective (or both if joint detection) across the val set
        - real/fake accuracy per transform type (if realfake task present)
        - transform accuracy per transform type (if transform task present)"""
    
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    metrics = [k for k in history[0] if k.startswith("acc_")]

    has_rf_by_transform = (
        "rf_by_transform" in history[0]
        and history[0]["rf_by_transform"]
        and "acc_realfake" in history[0]
    )
    has_tf_by_transform = (
        "tf_by_transform" in history[0]
        and history[0]["tf_by_transform"]
        and "acc_transform" in history[0]
    )

    transforms_rf = list(history[0]["rf_by_transform"].keys()) if has_rf_by_transform else []
    transforms_tf = list(history[0]["tf_by_transform"].keys()) if has_tf_by_transform else []

    n_plots = 1 + len(metrics) + (1 if has_rf_by_transform else 0) + (1 if has_tf_by_transform else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    axes = axes if n_plots > 1 else [axes]

    colors_tf = {"original": "steelblue", "transfer": "darkorange", "redigital": "mediumpurple"}
    ax_idx = 0

    # loss
    axes[ax_idx].plot(epochs, train_loss, marker="o")
    axes[ax_idx].set_title("Train loss")
    axes[ax_idx].set_xlabel("Epoch")
    axes[ax_idx].set_ylabel("Loss")
    axes[ax_idx].grid(True)
    ax_idx += 1

    # Global accuracies
    colors_acc = {"acc_realfake": "green", "acc_transform": "tomato"}
    for metric in metrics:
        values = [h[metric] for h in history]
        axes[ax_idx].plot(epochs, values, marker="o", color=colors_acc.get(metric, "gray"))
        axes[ax_idx].set_title(metric)
        axes[ax_idx].set_xlabel("Epoch")
        axes[ax_idx].set_ylabel("Accuracy")
        axes[ax_idx].set_ylim(0, 1)
        axes[ax_idx].grid(True)
        ax_idx += 1

    # RF accuracy per transformation
    if has_rf_by_transform:
        for t in transforms_rf:
            values = [h["rf_by_transform"].get(t) for h in history]
            axes[ax_idx].plot(epochs, values, marker="o", color=colors_tf.get(t, "gray"), label=t)
        axes[ax_idx].set_title("Real/fake acc by transform")
        axes[ax_idx].set_xlabel("Epoch")
        axes[ax_idx].set_ylabel("Accuracy")
        axes[ax_idx].set_ylim(0, 1)
        axes[ax_idx].legend()
        axes[ax_idx].grid(True)
        ax_idx += 1

    # Transform accuracy per transformation
    if has_tf_by_transform:
        for t in transforms_tf:
            values = [h["tf_by_transform"].get(t) for h in history]
            axes[ax_idx].plot(epochs, values, marker="o", color=colors_tf.get(t, "gray"), label=t)
        axes[ax_idx].set_title("Transform acc by transform")
        axes[ax_idx].set_xlabel("Epoch")
        axes[ax_idx].set_ylabel("Accuracy")
        axes[ax_idx].set_ylim(0, 1)
        axes[ax_idx].legend()
        axes[ax_idx].grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()