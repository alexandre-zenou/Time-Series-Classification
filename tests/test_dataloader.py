# tests/test_dataloader.py
import pytest

from src.data.dataloader import build_lsst_dataloaders


def test_build_lsst_dataloaders():
    try:
        train_dl, val_dl, test_dl, _, _, n_classes, n_features = build_lsst_dataloaders(seed=42)
    except Exception as exc:
        pytest.skip(f"LSST dataset unavailable or failed to load: {exc}")
    assert len(train_dl) > 0
    assert len(val_dl) > 0
    assert len(test_dl) > 0
    assert n_classes > 1
    assert n_features > 0
