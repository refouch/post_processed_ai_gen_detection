"""Evaluation module and metrics"""

import torch
from collections import defaultdict
from tqdm.auto import tqdm


@torch.no_grad()
def evaluate(model, loader, device):
    """Retourne accuracies globales + breakdown par transform et par contexte."""
    model.eval()
    is_multi = not hasattr(model, "task")

    correct_rf = total_rf = 0
    correct_tf = total_tf = 0

    rf_by_transform = defaultdict(lambda: [0, 0])
    rf_by_context   = defaultdict(lambda: [0, 0])
    tf_by_transform = defaultdict(lambda: [0, 0])  

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

        # Real/Fake task
        if pred_rf is not None:
            ok = (pred_rf.cpu() == y_rf)
            correct_rf += ok.sum().item(); total_rf += len(ok)
            for i in range(len(ok)):
                rf_by_transform[tf_str[i]][0] += ok[i].item()
                rf_by_transform[tf_str[i]][1] += 1
                if ctx_str[i] not in ("none", ""):
                    rf_by_context[ctx_str[i]][0] += ok[i].item()
                    rf_by_context[ctx_str[i]][1] += 1

        # Transform type task
        if pred_tf is not None:
            ok = (pred_tf.cpu() == y_tf)
            correct_tf += ok.sum().item(); total_tf += len(ok)
            for i in range(len(ok)):                
                tf_by_transform[tf_str[i]][0] += ok[i].item()
                tf_by_transform[tf_str[i]][1] += 1

    res = {}
    if total_rf: res["acc_realfake"]    = correct_rf / total_rf
    if total_tf: res["acc_transform"]   = correct_tf / total_tf
    res["rf_by_transform"] = {k: c/t for k,(c,t) in rf_by_transform.items()} if total_rf else {}
    res["rf_by_context"]   = {k: c/t for k,(c,t) in rf_by_context.items()}   if total_rf else {}
    res["tf_by_transform"] = {k: c/t for k,(c,t) in tf_by_transform.items()} if total_tf else {}
    
    return res