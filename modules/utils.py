"""Useful functions"""

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


def plot_history(history, title="Training history", save_path=None):
    """Function to plot the training history. Produces 3 or 4 plots:
        - Training loss
        - Accuracy to the objective (or both if joint detection) across the val set
        - real/fake accuray per transform type"""

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    metrics = [k for k in history[0] if k.startswith("acc_")]

    has_by_transform = (
        "rf_by_transform" in history[0]
        and history[0]["rf_by_transform"]           
        and "acc_realfake" in history[0]            
    )
    transforms = list(history[0]["rf_by_transform"].keys()) if has_by_transform else []

    n_plots = 1 + len(metrics) + (1 if has_by_transform else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    axes = axes if n_plots > 1 else [axes]

    # loss
    axes[0].plot(epochs, train_loss, marker="o")
    axes[0].set_title("Train loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    # accuracies
    colors_acc = {"acc_realfake": "green", "acc_transform": "tomato"}
    for ax, metric in zip(axes[1:], metrics):
        values = [h[metric] for h in history]
        ax.plot(epochs, values, marker="o", color=colors_acc.get(metric, "gray"))
        ax.set_title(metric)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.grid(True)

    # real/fake by tranformation
    if has_by_transform:
        ax = axes[-1]
        colors_tf = {"original": "steelblue", "transfer": "darkorange", "redigital": "mediumpurple"}
        for t in transforms:
            values = [h["rf_by_transform"].get(t) for h in history]
            ax.plot(epochs, values, marker="o", color=colors_tf.get(t, "gray"), label=t)
        ax.set_title("Real/fake acc by transform")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
