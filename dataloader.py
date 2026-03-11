from sklearn.calibration import LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset
from tslearn.datasets import UCR_UEA_datasets
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# FONCTIONS PARTAGÉES POUR COMPARAISON JUSTE
LSST_WINDOW = 36


def pad_truncate(X, tlen):
    """Pad ou truncate à longueur tlen (comme dans adapting_to_classification.py)"""
    out = []
    for arr in X:
        T = arr.shape[0]
        if T > tlen:
            out.append(arr[:tlen])
        else:
            pad = np.zeros((tlen - T, arr.shape[1]), dtype=arr.dtype)
            out.append(np.vstack((arr, pad)))
    return np.stack(out)


def build_lsst_dataloaders(seed=42, batch_size=32):
    """Dataloader unifié : normalisation, padding window=36, split stratify 80/20 (identique aux deux scripts)"""
    ds = UCR_UEA_datasets()
    X_train, y_train, X_test, y_test = ds.load_dataset("LSST")

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    n_classes = len(le.classes_)
    n_features = X_train.shape[2]

    # Padding/truncation à window=36 (comme IndPatchTST)
    X_train = pad_truncate(X_train, LSST_WINDOW)
    X_test = pad_truncate(X_test, LSST_WINDOW)

    # Normalisation StandardScaler (fit sur train)
    scaler = StandardScaler()
    X_train = (
        scaler.fit_transform(X_train.reshape(-1, n_features))
        .reshape(X_train.shape)
        .astype(np.float32)
    )
    X_test = (
        scaler.transform(X_test.reshape(-1, n_features))
        .reshape(X_test.shape)
        .astype(np.float32)
    )

    # Split train/val stratify (random_state=seed pour reproductibilité)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train_enc, test_size=0.2, stratify=y_train_enc, random_state=seed
    )

    # Tensors et DataLoaders (batch=32, drop_last=True pour train comme dans adapting)
    train_ds = TensorDataset(
        torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long()
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long()
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test).float(), torch.from_numpy(y_test_enc).long()
    )

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_dl, val_dl, test_dl, scaler, le, n_classes, n_features
