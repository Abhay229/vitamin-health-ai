"""Reproducible, leakage-safe training for Vitamin D level prediction."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.preprocessing import (
    DATA_PATH,
    PROJECT_ROOT,
    RANDOM_STATE,
    build_preprocessor,
    load_dataset,
    split_features_target,
)


MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def build_models(features: pd.DataFrame) -> dict[str, Pipeline]:
    """Build comparable pipelines with preprocessing fitted within each CV fold."""
    estimators = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=60, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=60,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=60,
            learning_rate=0.05,
            depth=6,
            loss_function="RMSE",
            random_seed=RANDOM_STATE,
            verbose=False,
            thread_count=1,
            allow_writing_files=False,
        ),
    }
    return {
        name: Pipeline([("preprocessor", build_preprocessor(features)), ("model", estimator)])
        for name, estimator in estimators.items()
    }


def metric_row(name: str, y_true: pd.Series, predictions: np.ndarray, cv: dict[str, np.ndarray]) -> dict[str, float | str]:
    return {
        "model": name,
        "cv_r2_mean": float(np.mean(cv["test_r2"])),
        "cv_r2_std": float(np.std(cv["test_r2"])),
        "cv_mae_mean": float(-np.mean(cv["test_neg_mae"])),
        "cv_rmse_mean": float(-np.mean(cv["test_neg_rmse"])),
        "test_r2": float(r2_score(y_true, predictions)),
        "test_mae": float(mean_absolute_error(y_true, predictions)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
    }


def train_and_save(data_path: Path | str = DATA_PATH) -> tuple[pd.DataFrame, str]:
    """Evaluate models on train-fold CV, evaluate once on holdout, then save the winner."""
    data = load_dataset(data_path)
    features, target = split_features_target(data)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=RANDOM_STATE
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"r2": "r2", "neg_mae": "neg_mean_absolute_error", "neg_rmse": "neg_root_mean_squared_error"}

    rows: list[dict[str, float | str]] = []
    fitted_models: dict[str, Pipeline] = {}
    holdout_predictions = pd.DataFrame({"actual_vitamin_d_ng_ml": y_test.to_numpy()})
    for name, pipeline in build_models(features).items():
        cv_scores = cross_validate(pipeline, x_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        rows.append(metric_row(name, y_test, predictions, cv_scores))
        holdout_predictions[f"{name.lower().replace(' ', '_')}_prediction"] = predictions
        fitted_models[name] = pipeline

    metrics = pd.DataFrame(rows).sort_values(["cv_rmse_mean", "cv_mae_mean"], ascending=True).reset_index(drop=True)
    best_name = str(metrics.iloc[0]["model"])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_models[best_name], MODELS_DIR / "vitamin_d_pipeline.joblib")
    metrics.to_csv(REPORTS_DIR / "model_metrics.csv", index=False)
    holdout_predictions.to_csv(REPORTS_DIR / "holdout_predictions.csv", index=False)
    metadata = {
        "best_model": best_name,
        "target": "VitaminD_Level_ng_ml",
        "feature_columns": features.columns.tolist(),
        "excluded_target_derived_columns": sorted(set(data.columns) - set(features.columns) - {"VitaminD_Level_ng_ml"}),
        "data_label": "Custom CSV supplied from the original repository; synthetic status is not documented.",
        "evaluation": "Five-fold shuffled CV on the training split; final metrics use an untouched 20% holdout.",
        "random_state": RANDOM_STATE,
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metrics, best_name


if __name__ == "__main__":
    metrics, best = train_and_save()
    print(f"Best model: {best}")
    print(metrics.to_string(index=False))
