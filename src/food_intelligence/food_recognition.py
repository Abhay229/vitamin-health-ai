"""Image-assisted food recognition behind a replaceable interface.

Design
------
* ``FoodRecognizer`` is a small protocol (``predict(image, top_k)``). Any model can be plugged in.
* The default implementation, ``HuggingFaceFoodRecognizer``, lazily loads a Food-101 image
  classifier (default ``nateraw/food``, a ViT fine-tuned on Food-101) with PyTorch + Transformers.
  Those packages are OPTIONAL. If they or the weights are unavailable, ``UnavailableRecognizer`` is
  used and manual food selection keeps working.
* Predictions are *suggestions*. Every prediction is flagged as requiring user confirmation, low
  confidence is highlighted, and nothing is ever added to the log automatically.
* Food-101 is a 101-class, single-dish, mostly Western-cuisine dataset. It does not cover most
  Indian dishes and cannot see portion size, ingredients or fortification. Recognition accuracy has
  NOT been measured for this project and no accuracy figure is claimed.
"""

from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.food_intelligence.nutrition_database import NutritionDatabase
from src.preprocessing import PROJECT_ROOT

LABEL_MAP_PATH = PROJECT_ROOT / "data" / "food101_label_map.csv"
DEFAULT_MODEL_ID = os.environ.get("FOOD_MODEL_ID", "nateraw/food")
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
ALLOWED_FORMATS = {"JPEG": "jpg/jpeg", "PNG": "png", "WEBP": "webp"}
LOW_CONFIDENCE_THRESHOLD = 0.50


class ImageValidationError(ValueError):
    """The uploaded file is not an acceptable image."""


class RecognizerUnavailableError(RuntimeError):
    """The recognition model or its dependencies cannot be used."""


def validate_image(data: bytes | None, max_bytes: int = MAX_IMAGE_BYTES):
    """Validate raw upload bytes and return a decoded RGB ``PIL.Image``.

    The content is decoded and verified; the file extension is not trusted.
    """
    from PIL import Image, UnidentifiedImageError  # Pillow ships with Streamlit

    if not data:
        raise ImageValidationError("The file is empty.")
    if len(data) > max_bytes:
        raise ImageValidationError(f"The image is larger than {max_bytes // (1024 * 1024)} MB.")
    try:
        probe = Image.open(io.BytesIO(data))
        image_format = probe.format
        probe.verify()  # integrity check; invalidates the probe object
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as error:
        raise ImageValidationError("The file could not be read as an image.") from error
    if image_format not in ALLOWED_FORMATS:
        raise ImageValidationError("Unsupported image type. Use JPG, PNG or WEBP.")
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ImageValidationError("The image has too many pixels.")
    if image.width < 32 or image.height < 32:
        raise ImageValidationError("The image is too small to analyse.")
    return image.convert("RGB")


@dataclass(frozen=True)
class RawPrediction:
    label: str
    confidence: float  # 0..1 as reported by the model


@dataclass(frozen=True)
class FoodPrediction:
    """A model label, optionally mapped to a record in the nutrition database."""

    label: str
    confidence: float | None
    food_id: str | None
    match_quality: str  # 'approximate' | 'none'
    note: str
    low_confidence: bool
    requires_confirmation: bool = True  # always true: predictions are never auto-accepted


@dataclass(frozen=True)
class RecognitionResult:
    predictions: tuple[FoodPrediction, ...]
    model_name: str
    available: bool
    message: str
    low_confidence: bool

    @property
    def mapped_food_ids(self) -> list[str]:
        seen: list[str] = []
        for p in self.predictions:
            if p.food_id and p.food_id not in seen:
                seen.append(p.food_id)
        return seen


class FoodRecognizer(Protocol):
    name: str

    def is_available(self) -> bool: ...

    def predict(self, image, top_k: int = 3) -> list[RawPrediction]: ...


class UnavailableRecognizer:
    """Used when the optional model cannot run. Always reports unavailability."""

    def __init__(self, reason: str, name: str = "unavailable"):
        self.name, self.reason = name, reason

    def is_available(self) -> bool:
        return False

    def predict(self, image, top_k: int = 3) -> list[RawPrediction]:
        raise RecognizerUnavailableError(self.reason)


class HuggingFaceFoodRecognizer:
    """Local Food-101 classifier through the Transformers image-classification pipeline.

    No paid API is used. Weights are downloaded once from the Hugging Face Hub when the user
    runs the download helper; afterwards ``local_files_only=True`` keeps inference offline.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, local_files_only: bool = True):
        self.name = model_id
        self._model_id = model_id
        self._local_only = local_files_only
        self._pipeline = None
        self._error: str | None = None

    def _load(self):
        if self._pipeline is not None or self._error is not None:
            return
        try:
            from transformers import pipeline  # optional dependency
        except ImportError:
            self._error = "Optional packages 'torch' and 'transformers' are not installed."
            return
        try:
            if self._local_only:  # keep inference offline once weights are cached
                import os
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
            self._pipeline = pipeline("image-classification", model=self._model_id)
        except Exception as error:  # weights missing, offline, incompatible versions ...
            self._error = f"Model weights for '{self._model_id}' could not be loaded ({type(error).__name__}). See README to download them."

    def is_available(self) -> bool:
        self._load()
        return self._pipeline is not None

    def predict(self, image, top_k: int = 3) -> list[RawPrediction]:
        self._load()
        if self._pipeline is None:
            raise RecognizerUnavailableError(self._error or "Model unavailable.")
        return [RawPrediction(str(r["label"]), float(r["score"])) for r in self._pipeline(image, top_k=top_k)]


def _normalize_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_").replace("-", "_")


def load_label_map(path: Path | str = LABEL_MAP_PATH) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return {_normalize_label(row["model_label"]): row for row in csv.DictReader(handle)}


def map_predictions(raw: list[RawPrediction], db: NutritionDatabase, label_map: dict[str, dict] | None = None,
                    threshold: float = LOW_CONFIDENCE_THRESHOLD, model_name: str = "custom") -> RecognitionResult:
    """Convert raw model output into confirmable suggestions mapped to the nutrition database."""
    label_map = load_label_map() if label_map is None else label_map
    predictions: list[FoodPrediction] = []
    for item in raw:
        if not 0.0 <= item.confidence <= 1.0:
            raise ValueError("Model confidence must be between 0 and 1.")
        mapping = label_map.get(_normalize_label(item.label))
        food_id = mapping["food_id"] if mapping and mapping["food_id"] in db else None
        predictions.append(FoodPrediction(
            label=item.label.replace("_", " "), confidence=item.confidence, food_id=food_id,
            match_quality=mapping["match_quality"] if food_id else "none",
            note=mapping["note"] if food_id else "No matching record in the nutrition database; choose a food manually.",
            low_confidence=item.confidence < threshold))
    top_low = not predictions or predictions[0].low_confidence
    if not predictions:
        message = "The model returned no suggestions. Please choose the food manually."
    elif top_low:
        message = "Low-confidence prediction. Please check carefully or choose the food manually."
    else:
        message = "Suggestions only. Please confirm or correct the food before it is logged."
    return RecognitionResult(tuple(predictions), model_name, True, message, top_low)


def recognize_food(image_bytes: bytes | None, recognizer: FoodRecognizer, db: NutritionDatabase, top_k: int = 3,
                   label_map: dict[str, dict] | None = None, threshold: float = LOW_CONFIDENCE_THRESHOLD) -> RecognitionResult:
    """Validate the upload, run the recognizer and return suggestions.

    Raises ``ImageValidationError`` for invalid input. If the recognizer is unavailable a
    result with ``available=False`` is returned so the UI can fall back to manual selection.
    """
    image = validate_image(image_bytes)
    if not recognizer.is_available():
        reason = getattr(recognizer, "reason", None) or getattr(recognizer, "_error", None) or "The recognition model is not available."
        return RecognitionResult((), recognizer.name, False, f"{reason} Please select the food manually.", True)
    try:
        raw = recognizer.predict(image, top_k=top_k)
    except RecognizerUnavailableError as error:
        return RecognitionResult((), recognizer.name, False, f"{error} Please select the food manually.", True)
    return map_predictions(raw, db, label_map, threshold, recognizer.name)


def apply_manual_correction(result: RecognitionResult, chosen_food_ids: list[str], db: NutritionDatabase) -> list[str]:
    """The user's confirmed selection always wins over the model. Unknown ids are rejected, duplicates dropped."""
    confirmed: list[str] = []
    for food_id in chosen_food_ids:
        if food_id not in db:
            raise KeyError(f"Unknown food_id: {food_id!r}")
        if food_id not in confirmed:
            confirmed.append(food_id)
    return confirmed


_default_recognizer: FoodRecognizer | None = None


def get_default_recognizer() -> FoodRecognizer:
    """Process-wide recognizer (lazy). Replace by passing another ``FoodRecognizer`` to ``recognize_food``."""
    global _default_recognizer
    if _default_recognizer is None:
        _default_recognizer = HuggingFaceFoodRecognizer()
    return _default_recognizer


def download_model(model_id: str = DEFAULT_MODEL_ID) -> None:  # pragma: no cover - explicit, user-initiated network use
    """One-off download of the weights: ``python -m src.food_intelligence.food_recognition``."""
    from transformers import pipeline

    pipeline("image-classification", model=model_id)
    print(f"Downloaded {model_id}. Inference now runs locally.")


if __name__ == "__main__":  # pragma: no cover
    download_model()
