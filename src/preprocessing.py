"""Dataset validation and leakage-safe preprocessing for Vitamin D regression."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "VitaminD_Dataset.csv"
TARGET_COLUMN = "VitaminD_Level_ng_ml"
RANDOM_STATE = 42

# These fields encode the target or are explicitly derived from it in the legacy project.
TARGET_DERIVED_COLUMNS = {
    "vitamind_category",
    "deficiency_status",
    "risk_score",
}


def load_dataset(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Read the supplied Vitamin D CSV and validate the regression target."""
    data = pd.read_csv(path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Dataset must contain {TARGET_COLUMN!r}.")
    target = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    if target.isna().any():
        raise ValueError(f"{TARGET_COLUMN} must contain only numeric, non-missing values.")
    return data


def _is_target_derived(column: str) -> bool:
    normalized = column.strip().lower()
    return normalized in TARGET_DERIVED_COLUMNS or normalized == TARGET_COLUMN.lower()


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return inputs and target while excluding target-derived columns from inputs."""
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Dataset must contain {TARGET_COLUMN!r}.")
    excluded = [column for column in data.columns if _is_target_derived(column)]
    features = data.drop(columns=excluded).copy()
    if features.empty:
        raise ValueError("No usable feature columns remain after leakage exclusion.")
    target = pd.to_numeric(data[TARGET_COLUMN], errors="raise").copy()
    return features, target


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Create an unfitted preprocessing step for numeric and categorical columns."""
    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]
    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")
