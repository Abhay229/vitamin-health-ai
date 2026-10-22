"""Load the saved Vitamin D pipeline and make validated single predictions."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing import PROJECT_ROOT


MODEL_PATH = PROJECT_ROOT / "models" / "vitamin_d_pipeline.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "metadata.json"


def load_model(model_path: Path | str = MODEL_PATH):
    if not Path(model_path).exists():
        raise FileNotFoundError("No trained model found. Run `python -m src.train` first.")
    return joblib.load(model_path)


def load_metadata(metadata_path: Path | str = METADATA_PATH) -> dict:
    return json.loads(Path(metadata_path).read_text(encoding="utf-8"))


def predict_vitamin_d(inputs: dict, model=None) -> float:
    """Predict serum Vitamin D level in ng/mL from one complete feature record."""
    metadata = load_metadata()
    missing = [column for column in metadata["feature_columns"] if column not in inputs]
    if missing:
        raise ValueError("Missing prediction fields: " + ", ".join(missing))
    record = pd.DataFrame([{column: inputs[column] for column in metadata["feature_columns"]}])
    pipeline = model if model is not None else load_model()
    return float(pipeline.predict(record)[0])


def vitamin_d_status(value: float) -> str:
    if value < 20:
        return "Deficient"
    if value <= 30:
        return "Insufficient"
    return "Sufficient"
