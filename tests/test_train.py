import pandas as pd

from src.train import build_models


def test_all_required_model_pipelines_are_created():
    features = pd.DataFrame({"Age": [20, 30, 40], "Gender": [0, 1, 0]})
    models = build_models(features)
    assert set(models) == {"Linear Regression", "Random Forest", "XGBoost", "CatBoost"}
    assert all(list(model.named_steps) == ["preprocessor", "model"] for model in models.values())
