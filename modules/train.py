"""Training loop for both unimodel and joint NN"""

import torch
import torch.nn as nn
from collections import defaultdict
from tqdm.auto import tqdm

def run_epoch(model, loader, optimizer, criterion, device,
              alpha=1.0, beta=1.0, train=True):
    """Une époque. Gère unimodal (model.task existe) et multi-tâche."""
    
    model.train() if train else model.eval()
    is_multi = not hasattr(model, "task")

    total_loss = 0.0
    n = 0
    torch.set_grad_enabled(train)

    pbar = tqdm(loader, desc="Train" if train else "Val", leave=False)
    for batch in pbar:
        img, y_rf, y_tf = batch[0], batch[1], batch[2]
        img = img.to(device)
        y_rf, y_tf = y_rf.to(device), y_tf.to(device)

        if is_multi:
            logits_rf, logits_tf = model(img)
            loss = alpha * criterion(logits_rf, y_rf) + beta * criterion(logits_tf, y_tf)
        else:
            y = y_rf if model.task == "realfake" else y_tf
            logits = model(img)
            loss = criterion(logits, y)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        bs = img.size(0)
        total_loss += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=f"{total_loss / n:.4f}")

    torch.set_grad_enabled(True)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device):
    """Retourne accuracies globales + breakdown par transform et par contexte."""
    model.eval()
    is_multi = not hasattr(model, "task")

    # compteurs : corrects / total, globaux et par groupe
    correct_rf = total_rf = 0
    correct_tf = total_tf = 0
    # real/fake accuracy ventilée par transformation et par contexte
    rf_by_transform = defaultdict(lambda: [0, 0])  # clé -> [corrects, total]
    rf_by_context   = defaultdict(lambda: [0, 0])

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
        else:  # transform
            pred_tf = model(img).argmax(1)
            pred_rf = None

        # --- tâche real/fake ---
        if pred_rf is not None:
            ok = (pred_rf.cpu() == y_rf)
            correct_rf += ok.sum().item(); total_rf += len(ok)
            for i in range(len(ok)):
                rf_by_transform[tf_str[i]][0] += ok[i].item()
                rf_by_transform[tf_str[i]][1] += 1
                rf_by_context[ctx_str[i]][0]  += ok[i].item()
                rf_by_context[ctx_str[i]][1]  += 1

        # --- tâche transform ---
        if pred_tf is not None:
            ok = (pred_tf.cpu() == y_tf)
            correct_tf += ok.sum().item(); total_tf += len(ok)

    res = {}
    if total_rf: res["acc_realfake"] = correct_rf / total_rf
    if total_tf: res["acc_transform"] = correct_tf / total_tf
    res["rf_by_transform"] = {k: c/t for k,(c,t) in rf_by_transform.items()} if total_rf else {}
    res["rf_by_context"]   = {k: c/t for k,(c,t) in rf_by_context.items()}   if total_rf else {}
    return res


def train_model(model, train_loader, val_loader, optimizer, criterion, device,
                epochs=10, alpha=1.0, beta=1.0, save_path=None):
    """Boucle complète avec suivi de la meilleure époque sur val."""
    best_val = -1.0
    best_state = None
    history = []

    epoch_bar = tqdm(range(1, epochs + 1), desc="Epochs")
    for epoch in epoch_bar:
        tr_loss = run_epoch(model, train_loader, optimizer, criterion, device,
                            alpha, beta, train=True)
        val = evaluate(model, val_loader, device)

        # critère de sélection : moyenne des accuracies disponibles
        accs = [v for k, v in val.items() if k.startswith("acc_")]
        val_score = sum(accs) / len(accs)

        epoch_bar.set_postfix({k: f"{v:.4f}" for k, v in val.items() if k.startswith("acc_")} | {"loss": f"{tr_loss:.4f}"})
        print(f"[ep {epoch:02d}] train_loss={tr_loss:.4f} | "
              + " | ".join(f"{k}={v:.4f}" for k, v in val.items() if k.startswith("acc_")))

        history.append({"epoch": epoch, "train_loss": tr_loss, **val})
        if val_score > best_val:
            best_val = val_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        if save_path is not None:
            torch.save(best_state, save_path)
            print(f"Model saved to {save_path}")
    return model, history