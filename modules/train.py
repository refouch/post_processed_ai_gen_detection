"""Training loop for both unimodel and joint NN"""

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from modules.evaulation import evaluate

def run_epoch(model, loader, optimizer, criterion, device, alpha=1.0, beta=1.0, train=True):
    """Train step for 1 epoch, handles the different losses bewteen unimodal and joint detection"""
    
    model.train() if train else model.eval()
    is_multi = not hasattr(model, "task")

    total_loss = 0.0
    n = 0
    torch.set_grad_enabled(train)

    pbar = tqdm(loader, desc="Train" if train else "Val", leave=False)
    for batch in pbar:
        img, y_rf, y_tf = batch[0], batch[1], batch[2] # Retrieve the labels for both
        img = img.to(device)
        y_rf, y_tf = y_rf.to(device), y_tf.to(device)

        if is_multi:
            logits_rf, logits_tf = model(img)
            loss = alpha * criterion(logits_rf, y_rf) + beta * criterion(logits_tf, y_tf) # Weighted loss for joint detection
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

def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs=10, alpha=1.0, beta=1.0, save_path=None):
    """Full training loop, with evaluation on the validation set"""

    best_val = -1.0
    best_state = None
    history = []

    epoch_bar = tqdm(range(1, epochs + 1), desc="Epochs")
    for epoch in epoch_bar:

        tr_loss = run_epoch(model, train_loader, optimizer, criterion, device,alpha, beta, train=True) # train
        val = evaluate(model, val_loader, device) # validate

        # Selection criterion: mean of available accuracies
        accs = [v for k, v in val.items() if k.startswith("acc_")]
        val_score = sum(accs) / len(accs)

        #tqdm stuff
        epoch_bar.set_postfix({k: f"{v:.4f}" for k, v in val.items() if k.startswith("acc_")} | {"loss": f"{tr_loss:.4f}"})
        print(f"[ep {epoch:02d}] train_loss={tr_loss:.4f} | "
              + " | ".join(f"{k}={v:.4f}" for k, v in val.items() if k.startswith("acc_")))

        history.append({"epoch": epoch, "train_loss": tr_loss, **val}) # keep track of the training history

        if val_score > best_val: # keep track of the best performing state
            best_val = val_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        if save_path is not None: # saving the weights to avoid re-training!
            torch.save(best_state, save_path)
            print(f"Model saved to {save_path}")
            
    return model, history