def test_imports():
    import src.models
    from src.models import CNNBaseline, IndPatchTST, IndPatchTSTClassifier
    from src.training import optuna_search
    from src.training import adapting_to_classification

    assert CNNBaseline
    assert IndPatchTST
    assert IndPatchTSTClassifier
    assert optuna_search
    assert adapting_to_classification
