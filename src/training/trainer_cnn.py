import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from src.data.dataloader import build_lsst_dataloaders
from src.models.cnn_baseline import CNNBaseline


# FONCTIONS D'ÉVAL/ENTRAÎNEMENT (adaptées pour stats)
def evaluate(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, f1


def train_one_model(
    model,
    train_loader,
    val_loader,
    max_epochs=50,
    patience=7,
    lr=1e-3,
    weight_decay=1e-4,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_val_f1 = -np.inf
    best_state = None
    patience_counter = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        val_acc, val_f1 = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch:02d} | Train loss: {avg_train_loss:.4f} | Val acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
        )
        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping à l'epoch {epoch} (best val F1 = {best_val_f1:.4f})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_f1


def hyperparam_search(train_loader, val_loader, n_features, n_classes, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    search_space = [
        {
            "n_filters1": 32,
            "n_filters2": 64,
            "n_filters3": 128,
            "n_filters4": 256,
            "dropout": 0.3,
            "lr": 1e-3,
            "weight_decay": 1e-4,
        },
        {
            "n_filters1": 64,
            "n_filters2": 128,
            "n_filters3": 256,
            "n_filters4": 512,
            "dropout": 0.5,
            "lr": 1e-3,
            "weight_decay": 1e-4,
        },
        {
            "n_filters1": 64,
            "n_filters2": 128,
            "n_filters3": 256,
            "n_filters4": 512,
            "dropout": 0.3,
            "lr": 5e-4,
            "weight_decay": 1e-4,
        },
        {
            "n_filters1": 64,
            "n_filters2": 128,
            "n_filters3": 256,
            "n_filters4": 512,
            "dropout": 0.3,
            "lr": 1e-3,
            "weight_decay": 5e-4,
        },
    ]
    best_config, best_model, best_val_f1 = None, None, -np.inf
    for i, cfg in enumerate(search_space, 1):
        print(f"\n=== Config {i}/{len(search_space)}: {cfg} ===")
        model = CNNBaseline(
            n_features=n_features,
            n_classes=n_classes,
            **{
                k: cfg[k]
                for k in [
                    "n_filters1",
                    "n_filters2",
                    "n_filters3",
                    "n_filters4",
                    "dropout",
                ]
            },
        )
        model, val_f1 = train_one_model(
            model,
            train_loader,
            val_loader,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            device=device,
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_config = cfg
            best_model = model
    print(f"\nMeilleure config: {best_config} avec val F1 = {best_val_f1:.4f}")
    return best_model, best_config, best_val_f1


def run_statistics_cnn(n_runs=15, base_seed=0):
    """15 runs avec seeds comme dans adapting_to_classification.py"""
    all_accs, all_f1s = [], []
    for i in range(n_runs):
        seed = base_seed + i
        print(f"\n=== RUN {i + 1}/{n_runs} (seed={seed}) ===")
        train_dl, val_dl, test_dl, _, _, n_classes, n_features = build_lsst_dataloaders(seed=seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best_model, best_cfg, best_val_f1 = hyperparam_search(
            train_dl, val_dl, n_features, n_classes, device
        )
        test_acc, test_f1 = evaluate(best_model, test_dl, device)
        print(f"Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")
        all_accs.append(test_acc)
        all_f1s.append(test_f1)
    print("\n=== STATISTIQUES (15 runs) ===")
    print(
        f"Acc: {np.mean(all_accs):.3f} ± {np.std(all_accs):.3f} (min={np.min(all_accs):.3f}, max={np.max(all_accs):.3f})"
    )
    print(f"F1:  {np.mean(all_f1s):.3f} ± {np.std(all_f1s):.3f}")
    return np.mean(all_accs), np.std(all_accs), np.mean(all_f1s), np.std(all_f1s)
