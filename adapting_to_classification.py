from transformer_pretraining import build_etth1_dataloaders, IndPatchTST
import torch.nn as nn
import torch


class PatchTSTForClassification(nn.Module):
    def __init__(
        self,
        backbone: IndPatchTST,  # ou nn.Module si tu veux rester générique
        d_model,
        n_classes,
    ):
        super().__init__()
        self.backbone = backbone
        self.d_model = d_model
        self.classifier = nn.Linear(self.d_model, n_classes)

        # (optionnel) si jamais tu remets revin=True plus tard
        if hasattr(self.backbone, "revin_layer") and self.backbone.revin:
            # pour classification, pas besoin de denorm, donc tu peux le couper
            self.backbone.revin_layer = nn.Identity()

    def forward(self, x):
        # x: (B, T, C) -> features (B, d_model) -> logits (B, n_classes)
        feats = self.backbone.forward_features(x)
        return self.classifier(feats)


def train_epoch_classification(model, dataloader, optimizer, criterion, device="cuda"):
    model.to(device)
    model.train()
    total_loss = 0.0
    for past, future in dataloader:
        past, future = past.to(device), future.to(device)
        optimizer.zero_grad()
        pred = model(past)  # pred: (batch, n_classes)
        loss = criterion(pred, future)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * past.size(0)
    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def eval_epoch_classification(model, dataloader, criterion, device="cuda"):
    model.to(device)
    model.eval()
    total_loss = 0.0
    for past, future in dataloader:
        past, future = past.to(device), future.to(device)
        pred = model(past)
        loss = criterion(pred, future)
        total_loss += loss.item() * past.size(0)
    return total_loss / len(dataloader.dataset)


def train_and_valid_loop(
    model, train_dl, valid_dl, optimizer, criterion, n_epochs, device="cuda"
):
    logs = {"train_loss": [], "valid_loss": []}
    print(model.__class__.__name__)
    for epoch in range(n_epochs):
        train_loss = train_epoch_classification(
            model, train_dl, optimizer, criterion, device=device
        )
        logs["train_loss"].append(train_loss)
        valid_loss = eval_epoch_classification(
            model, valid_dl, criterion, device=device
        )
        logs["valid_loss"].append(valid_loss)
        print(f"Epoch {epoch:02d} | train={train_loss:.4f} | valid={valid_loss:.4f}")
    return logs


if __name__ == "__main__":
    import torch
    import numpy as np
    from tslearn.datasets import UCR_UEA_datasets
    from sklearn.preprocessing import LabelEncoder
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    window = 36
    horizon = 6
    train_dl, valid_dl, input_dim = build_etth1_dataloaders(
        "..\\data\\ETTh1.csv", window=window, horizon=horizon
    )
    forecast_model = IndPatchTST(
        window,
        horizon,
        num_features=train_dl.dataset.feats.shape[1],
        patch_len=2,
        stride=2,
        revin=False,
    )
    forecast_model = forecast_model.to(device)
    forecast_model.load_state_dict(torch.load("models\\patchtst_etth1.pth"))
    ds = UCR_UEA_datasets()
    X_train, y_train, X_test, y_test = ds.load_dataset("LSST")

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    n_classes = len(le.classes_)
    n_features = X_train.shape[2]

    # >>>>>> AJOUT ICI : pad / truncate à window=96 <<<<<<
    TARGET_LEN = window  # 96

    def pad_truncate_numpy(X, target_len):
        out = []
        for arr in X:
            T = arr.shape[0]
            if T >= target_len:
                out.append(arr[:target_len])
            else:
                pad = np.zeros((target_len - T, arr.shape[1]), dtype=arr.dtype)
                out.append(np.vstack([arr, pad]))
        return np.stack(out, axis=0)

    X_train = pad_truncate_numpy(X_train, TARGET_LEN)
    X_test = pad_truncate_numpy(X_test, TARGET_LEN)
    print("LSST shapes après pad/truncate:", X_train.shape, X_test.shape)
    # doit afficher (N_train, 96, 6) et (N_test, 96, 6)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    n_classes = len(le.classes_)
    n_features = X_train.shape[2]

    # 3) Construire TensorDatasets + DataLoaders (split train/val 80/20)
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train_enc).long()
    X_test_t = torch.from_numpy(X_test).float()
    y_test_t = torch.from_numpy(y_test_enc).long()

    n_train = len(X_train_t)
    n_val = int(0.2 * n_train)

    X_tr, y_tr = X_train_t[: n_train - n_val], y_train_t[: n_train - n_val]
    X_val, y_val = X_train_t[n_train - n_val :], y_train_t[n_train - n_val :]

    train_ds = TensorDataset(X_tr, y_tr)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, drop_last=False)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, drop_last=False)

    classification_model = PatchTSTForClassification(
        backbone=forecast_model,
        d_model=forecast_model.d_model,
        n_classes=n_classes,
    ).to(device)

    # Geler le backbone pré-entraîné
    for p in classification_model.backbone.parameters():
        p.requires_grad = False

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classification_model.classifier.parameters(), lr=1e-4)
    n_epochs = 10
    best_val_f1 = 0.0
    train_and_valid_loop(
        classification_model,
        train_dl,
        val_dl,
        optimizer,
        loss_fn,
        n_epochs,
        device=device,
    )
    from baseline import evaluate

    # 5) Évaluation finale sur test
    test_acc, test_f1 = evaluate(classification_model, test_dl, device)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test F1 (macro): {test_f1:.4f}")
