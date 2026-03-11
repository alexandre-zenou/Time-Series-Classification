import optuna
import torch
import torch.nn as nn

from src.models.indpatchtst_classifier import IndPatchTSTClassifier
from src.models.indpatchtst import build_model_from_config
from src.training.indpatchtst_clf_utils import _build_clf_model
from src.training.train_indpatchtst_class import eval_epoch as eval_epoch_clf
from src.training.train_indpatchtst_class import (
    train_epoch as train_epoch_clf,
)
from src.training.train_indpatchtst_reg import eval_epoch, train_epoch


def objective(trial, train_dl, valid_dl, window, horizon, device, max_epochs=20):
    """
    Contrainte clé : n_heads doit diviser d_model.
    Espace restreint pour RTX 4060 (8 GB VRAM).
    """
    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    valid_heads = [h for h in [1, 2, 4, 8] if d_model % h == 0]
    n_heads = trial.suggest_categorical("n_heads", valid_heads)

    patch_len = trial.suggest_int("patch_len", 3, window // 3)
    stride = trial.suggest_int("stride", 1, max(1, window // 8))
    if patch_len >= window:
        raise optuna.TrialPruned()

    config = {
        "patch_len": patch_len,
        "stride": stride,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": trial.suggest_int("n_layers", 2, 6),
        "d_ff": trial.suggest_categorical("d_ff", [256, 512, 1024]),
        "dropout": trial.suggest_float("dropout", 0.05, 0.3),
        "revin": trial.suggest_categorical("revin", [True, False]),
        "lr": trial.suggest_float("lr", 5e-5, 5e-3, log=True),
    }

    num_features = train_dl.dataset.tensors[0].shape[2]
    model = build_model_from_config(config, num_features, window, horizon).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_valid_loss, patience_counter, patience = float("inf"), 0, 4
    for epoch in range(max_epochs):
        train_epoch(model, train_dl, optimizer, criterion, device)
        valid_loss = eval_epoch(model, valid_dl, criterion, device)
        trial.report(valid_loss, epoch)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_valid_loss


def bayesian_search(train_dl, valid_dl, window, horizon, device, n_trials=30, max_epochs=20):
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=4),
    )
    study.optimize(
        lambda trial: objective(trial, train_dl, valid_dl, window, horizon, device, max_epochs),
        n_trials=n_trials,
    )
    print("\n=== Meilleure config bayésienne ===")
    print(study.best_params)
    print(f"Meilleure valid loss: {study.best_value:.4f}")
    return study.best_params, study.best_value


# Classification


def objective_head_only(
    trial,
    train_dl,
    val_dl,
    device,
    window,
    n_features,
    n_classes,
    backbone_config,
    scaler_amp,
):
    """
    Stratégie B — Head Only.
    Backbone entièrement gelé : seule la tête est entraînée.
    Hyperparamètres tunés : hidden_dim, dropout_clf, lr_head, weight_decay.
    lr_backbone n'est PAS tuné ici car il n'est pas utilisé dans cette stratégie.
    """
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    dropout_clf = trial.suggest_float("dropout_clf", 0.2, 0.5)
    lr_head = trial.suggest_float("lr_head", 1e-4, 3e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)

    model = _build_clf_model(
        window, n_features, n_classes, backbone_config, hidden_dim, dropout_clf, device
    )
    model.freeze_all_backbone()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.classifier.parameters(), lr=lr_head, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
    best_val_acc = 0.0
    for epoch in range(20):
        train_epoch_clf(
            model, train_dl, opt, criterion, device, augment=True, scaler_amp=scaler_amp
        )
        _, vl_acc = eval_epoch_clf(model, val_dl, criterion, device)
        sch.step()
        trial.report(vl_acc, epoch)
        if trial.should_prune():
            raise __import__("optuna").TrialPruned()
        best_val_acc = max(best_val_acc, vl_acc)
    return best_val_acc


def objective_late_enc(
    trial,
    train_dl,
    val_dl,
    device,
    window,
    n_features,
    n_classes,
    backbone_config,
    scaler_amp,
):
    """
    Stratégie C — Late Encoders + Head.
    Les 2 dernières couches transformer + tête sont dégelées.
    Hyperparamètres tunés : lr_late (late encoders), lr_head, hidden_dim,
    dropout_clf, weight_decay.
    lr_late est distinct de lr_backbone car seules les late layers bougent.
    """
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    dropout_clf = trial.suggest_float("dropout_clf", 0.2, 0.5)
    lr_late = trial.suggest_float("lr_late", 1e-5, 5e-4, log=True)
    lr_head = trial.suggest_float("lr_head", 1e-4, 3e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)

    model = _build_clf_model(
        window, n_features, n_classes, backbone_config, hidden_dim, dropout_clf, device
    )
    model.unfreeze_late_encoders()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(
        [
            {"params": model.group_enc_late.parameters(), "lr": lr_late},
            {"params": model.classifier.parameters(), "lr": lr_head},
        ],
        weight_decay=wd,
    )
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
    best_val_acc = 0.0
    for epoch in range(20):
        train_epoch_clf(
            model, train_dl, opt, criterion, device, augment=True, scaler_amp=scaler_amp
        )
        _, vl_acc = eval_epoch_clf(model, val_dl, criterion, device)
        sch.step()
        trial.report(vl_acc, epoch)
        if trial.should_prune():
            raise __import__("optuna").TrialPruned()
        best_val_acc = max(best_val_acc, vl_acc)
    return best_val_acc


def objective_full_tune(
    trial,
    train_dl,
    val_dl,
    device,
    window,
    n_features,
    n_classes,
    backbone_config,
    scaler_amp,
):
    """
    Stratégie D — Full Fine-tune avec LR différenciés.
    Tout le modèle est dégelé. lr_backbone ≪ lr_head pour protéger
    les features pré-apprises du catastrophic forgetting.
    Hyperparamètres tunés : lr_backbone, lr_head, hidden_dim, dropout_clf, wd.
    """
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    dropout_clf = trial.suggest_float("dropout_clf", 0.2, 0.5)
    lr_backbone = trial.suggest_float("lr_backbone", 1e-5, 3e-4, log=True)
    lr_head = trial.suggest_float("lr_head", 1e-4, 3e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)

    model = _build_clf_model(
        window, n_features, n_classes, backbone_config, hidden_dim, dropout_clf, device
    )
    model.unfreeze_all()

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": lr_backbone},
            {"params": model.classifier.parameters(), "lr": lr_head},
        ],
        weight_decay=wd,
    )
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
    best_val_acc = 0.0
    for epoch in range(20):
        train_epoch_clf(
            model,
            train_dl,
            opt,
            criterion,
            device,
            augment=False,
            scaler_amp=scaler_amp,
        )
        _, vl_acc = eval_epoch_clf(model, val_dl, criterion, device)
        sch.step()
        trial.report(vl_acc, epoch)
        if trial.should_prune():
            raise __import__("optuna").TrialPruned()
        best_val_acc = max(best_val_acc, vl_acc)
    return best_val_acc


def objective_scratch(
    trial,
    train_dl,
    val_dl,
    device,
    window,
    n_features,
    n_classes,
    scaler_amp,
):
    """
    Recherche Optuna dédiée au modèle From Scratch.

    Explore simultanément la config backbone ET les hyperparamètres
    d'entraînement, directement sur la tâche de classification LSST.
    C'est nécessaire car la config optimale pour la régression ETTh1
    (minimiser MSE sur séries météo) n'est pas forcément optimale pour
    classifier des objets astronomiques multivariés.
    """
    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    valid_heads = [h for h in [1, 2, 4, 8] if d_model % h == 0]
    n_heads = trial.suggest_categorical("n_heads", valid_heads)
    patch_len = trial.suggest_int("patch_len", 3, window // 3)
    stride = trial.suggest_int("stride", 1, max(1, window // 8))
    if patch_len >= window:
        raise __import__("optuna").TrialPruned()

    scratch_config = {
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": trial.suggest_int("n_layers", 2, 6),
        "d_ff": trial.suggest_categorical("d_ff", [256, 512, 1024]),
        "dropout": trial.suggest_float("dropout", 0.05, 0.4),
        "revin": False,
        "patch_len": patch_len,
        "stride": stride,
    }

    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    dropout_clf = trial.suggest_float("dropout_clf", 0.2, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    wd = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)

    model = IndPatchTSTClassifier(
        window,
        n_features,
        n_classes,
        scratch_config,
        pretrained_model_path=None,
    ).to(device)

    model.classifier = nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout_clf),
        nn.Linear(hidden_dim, n_classes),
    ).to(device)

    model.unfreeze_all()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
    best_val_acc = 0.0

    for epoch in range(20):
        train_epoch_clf(
            model, train_dl, opt, criterion, device, augment=True, scaler_amp=scaler_amp
        )
        _, vl_acc = eval_epoch_clf(model, val_dl, criterion, device)
        sch.step()
        trial.report(vl_acc, epoch)
        if trial.should_prune():
            raise __import__("optuna").TrialPruned()
        best_val_acc = max(best_val_acc, vl_acc)

    return best_val_acc
