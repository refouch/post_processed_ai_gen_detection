"""Evaluation module and metrics"""

import os
import torch
import numpy as np
from collections import defaultdict
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


@torch.no_grad()
def evaluate(model, loader, device, name=None):
    """Returns global accuracies + per-transform / per-context breakdowns.

    Args:
        name: if provided, saves confusion-matrix plots to
              plots/confusion_matrixes/{name}_{task}.png
    """
    model.eval()
    is_multi = not hasattr(model, "task")

    correct_rf = total_rf = 0
    correct_tf = total_tf = 0

    rf_by_transform = defaultdict(lambda: [0, 0])
    rf_by_context   = defaultdict(lambda: [0, 0])
    tf_by_transform = defaultdict(lambda: [0, 0])

    all_pred_rf, all_true_rf = [], []
    all_pred_tf, all_true_tf = [], []

    for batch in tqdm(loader, desc="Evaluate", leave=False):
        img, y_rf, y_tf, tf_str, ctx_str = batch
        img = img.to(device)
        y_rf_d, y_tf_d = y_rf.to(device), y_tf.to(device)

        if is_multi:
            logits_rf, logits_tf = model(img)
            pred_rf = logits_rf.argmax(1)
            pred_tf = logits_tf.argmax(1)
        elif model.task == "realfake":
            pred_rf = model(img).argmax(1)
            pred_tf = None
        else:
            pred_tf = model(img).argmax(1)
            pred_rf = None

        if pred_rf is not None:
            ok = (pred_rf.cpu() == y_rf)
            correct_rf += ok.sum().item()
            total_rf   += len(ok)
            all_pred_rf.extend(pred_rf.cpu().tolist())
            all_true_rf.extend(y_rf.tolist())
            for i in range(len(ok)):
                tf  = tf_str[i]
                ctx = ctx_str[i]
                if tf and tf != 'nan':
                    rf_by_transform[tf][0] += ok[i].item()
                    rf_by_transform[tf][1] += 1
                if ctx and ctx not in ("none", "", "nan"):
                    rf_by_context[ctx][0] += ok[i].item()
                    rf_by_context[ctx][1] += 1

        if pred_tf is not None:
            ok = (pred_tf.cpu() == y_tf)
            correct_tf += ok.sum().item()
            total_tf   += len(ok)
            all_pred_tf.extend(pred_tf.cpu().tolist())
            all_true_tf.extend(y_tf.tolist())
            for i in range(len(ok)):
                tf = tf_str[i]
                if tf and tf != 'nan':
                    tf_by_transform[tf][0] += ok[i].item()
                    tf_by_transform[tf][1] += 1

    res = {}
    if total_rf:
        res["acc_realfake"]    = correct_rf / total_rf
    if total_tf:
        res["acc_transform"]   = correct_tf / total_tf
    res["rf_by_transform"] = {k: c/t for k, (c, t) in rf_by_transform.items()} if total_rf else {}
    res["rf_by_context"]   = {k: c/t for k, (c, t) in rf_by_context.items()}   if total_rf else {}
    res["tf_by_transform"] = {k: c/t for k, (c, t) in tf_by_transform.items()} if total_tf else {}

    if name:
        _save_confusion_matrices(name, all_true_rf, all_pred_rf, all_true_tf, all_pred_tf)

    return res


def _save_confusion_matrices(name, true_rf, pred_rf, true_tf, pred_tf):
    """Compute and save normalised confusion matrices with raw counts in a separate folder."""

    os.makedirs('plots/confusion_matrixes', exist_ok=True)

    tasks = []
    if true_rf:
        tasks.append((true_rf, pred_rf, ['Real', 'AI'],
                      'realfake', 'Real / Fake detection'))
    if true_tf:
        tasks.append((true_tf, pred_tf, ['Original', 'Transfer', 'Redigital'],
                      'transform', 'Transform-type identification'))

    for true, pred, labels, tag, title in tasks:
        n = len(labels)
        cm = np.zeros((n, n), dtype=int)
        for t, p in zip(true, pred):
            cm[t][p] += 1

        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        annot = np.array([
            [f'{cm_norm[i, j]:.1%}\n(n={cm[i, j]:,})' for j in range(n)]
            for i in range(n)
        ])

        fig, ax = plt.subplots(figsize=(n * 2.8 + 0.5, n * 2.6))
        sns.heatmap(
            cm_norm, annot=annot, fmt='', cmap='Blues',
            xticklabels=labels, yticklabels=labels,
            vmin=0, vmax=1,
            linewidths=1.0, linecolor='white',
            ax=ax,
            cbar_kws={'label': 'Row-normalised proportion', 'shrink': 0.75},
            annot_kws={'size': 12, 'weight': 'bold'},
        )
        ax.set_xlabel('Predicted label', fontsize=12, labelpad=10)
        ax.set_ylabel('True label',      fontsize=12, labelpad=10)
        ax.set_title(f'{title}\n{name}', fontsize=13, fontweight='bold', pad=14)
        ax.tick_params(axis='both', labelsize=11)

        plt.tight_layout()
        fig.savefig(f'plots/confusion_matrixes/{name}_{tag}.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  saved plots/confusion_matrixes/{name}_{tag}.png')