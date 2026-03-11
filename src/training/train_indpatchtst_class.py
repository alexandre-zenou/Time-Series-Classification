from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch.nn as nn
from src.training.indpatchtst_clf_utils import augment_batch
import torch


def train_epoch(model, dl, optimizer, criterion, device, augment=False, scaler_amp=None):
    model.train()
    total_loss, correct, n_samples = 0.0, 0, 0

    for x, y in dl:
        x, y = x.to(device), y.to(device)
        if augment:
            x = augment_batch(x)

        optimizer.zero_grad()

        if scaler_amp is not None:
            with torch.amp.autocast(device_type="cuda"):
                pred = model(x)
                loss = criterion(pred, y)
            # Si la loss est NaN/inf (overflow float16), scaler skip le step
            # mais on doit aussi ne PAS accumuler dans total_loss
            if not torch.isfinite(loss):
                scaler_amp.update()  # met à jour le scale factor (le réduit)
                continue
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            pred = model(x)
            loss = criterion(pred, y)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (pred.argmax(1) == y).sum().item()
        n_samples += x.size(0)

    if n_samples == 0:
        return float("nan"), 0.0
    return total_loss / n_samples, correct / n_samples


@torch.no_grad()
def eval_epoch(model, dl, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total_loss += criterion(pred, y).item() * x.size(0)
        correct += (pred.argmax(1) == y).sum().item()
    return total_loss / len(dl.dataset), correct / len(dl.dataset)


def train_loop(
    model,
    train_dl,
    val_dl,
    optimizer,
    criterion,
    n_epochs,
    device,
    scheduler=None,
    augment=False,
    patience=8,
    scaler_amp=None,
):
    logs = {"train_loss": [], "train_acc": [], "valid_loss": [], "valid_acc": []}
    best_val_acc, patience_ctr = -1.0, 0
    best_state = None

    for epoch in range(n_epochs):
        tr_loss, tr_acc = train_epoch(
            model, train_dl, optimizer, criterion, device, augment, scaler_amp
        )
        vl_loss, vl_acc = eval_epoch(model, val_dl, criterion, device)

        if scheduler:
            scheduler.step()

        logs["train_loss"].append(tr_loss)
        logs["train_acc"].append(tr_acc)
        logs["valid_loss"].append(vl_loss)
        logs["valid_acc"].append(vl_acc)
        print(
            f"    Epoch {epoch + 1:02d}/{n_epochs} | "
            f"TrL={tr_loss:.4f} TrA={tr_acc:.3f} | "
            f"VlL={vl_loss:.4f} VlA={vl_acc:.3f}"
        )

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"    ⏹ Early stopping | meilleur VlA={best_val_acc:.3f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"    ✅ Meilleur modèle restauré (VlA={best_val_acc:.3f})")

    return logs


@torch.no_grad()
def evaluate(model, test_dl, device):
    model.eval()
    all_pred, all_y = [], []
    for x, y in test_dl:
        all_pred.extend(model(x.to(device)).argmax(1).cpu().numpy())
        all_y.extend(y.numpy())
    all_pred, all_y = np.array(all_pred), np.array(all_y)
    return accuracy_score(all_y, all_pred), f1_score(all_y, all_pred, average="macro")
