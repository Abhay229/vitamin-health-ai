import io

import pytest
from PIL import Image

from src.food_intelligence.food_recognition import (
    HuggingFaceFoodRecognizer, ImageValidationError, RawPrediction, RecognizerUnavailableError, UnavailableRecognizer,
    apply_manual_correction, load_label_map, map_predictions, recognize_food, validate_image,
)


def png_bytes(size=(64, 64), fmt="PNG"):
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 120, 40)).save(buf, fmt)
    return buf.getvalue()


class FakeRecognizer:
    name = "fake"

    def __init__(self, preds):
        self.preds = preds

    def is_available(self):
        return True

    def predict(self, image, top_k=3):
        assert image.mode == "RGB"
        return self.preds[:top_k]


def test_valid_image_accepted():
    assert validate_image(png_bytes()).size == (64, 64)
    assert validate_image(png_bytes(fmt="JPEG")).mode == "RGB"


@pytest.mark.parametrize("data", [b"", None, b"not an image", b"%PDF-1.4 fake", png_bytes()[:30]])
def test_invalid_images_rejected(data):
    with pytest.raises(ImageValidationError):
        validate_image(data)


def test_oversized_and_tiny_and_unsupported_rejected():
    with pytest.raises(ImageValidationError):
        validate_image(png_bytes(), max_bytes=10)
    with pytest.raises(ImageValidationError):
        validate_image(png_bytes(size=(8, 8)))
    with pytest.raises(ImageValidationError):
        validate_image(png_bytes(fmt="GIF"))


def test_label_map_only_references_existing_foods(db):
    assert all(row["food_id"] in db for row in load_label_map().values())


def test_predictions_always_require_confirmation_and_map_to_db(db):
    result = recognize_food(png_bytes(), FakeRecognizer([RawPrediction("omelette", 0.9), RawPrediction("pizza", 0.05)]), db)
    assert result.available and not result.low_confidence
    assert all(p.requires_confirmation for p in result.predictions)
    assert result.predictions[0].food_id == "egg_boiled" and result.predictions[1].food_id is None
    assert result.mapped_food_ids == ["egg_boiled"]


def test_low_confidence_is_flagged(db):
    result = recognize_food(png_bytes(), FakeRecognizer([RawPrediction("omelette", 0.31)]), db)
    assert result.low_confidence and result.predictions[0].low_confidence and "Low-confidence" in result.message


def test_bad_confidence_rejected(db):
    with pytest.raises(ValueError):
        map_predictions([RawPrediction("omelette", 1.5)], db)


def test_missing_model_weights_falls_back_to_manual(db):
    result = recognize_food(png_bytes(), UnavailableRecognizer("weights missing"), db)
    assert not result.available and result.predictions == () and "manually" in result.message


def test_real_recognizer_degrades_gracefully_without_weights(db):
    rec = HuggingFaceFoodRecognizer(model_id="does-not-exist/none", local_files_only=True)
    assert rec.is_available() is False  # no torch/transformers or no cached weights in the test environment
    with pytest.raises(RecognizerUnavailableError):
        rec.predict(validate_image(png_bytes()))
    assert not recognize_food(png_bytes(), rec, db).available


def test_invalid_upload_raises_before_model_runs(db):
    class Boom(FakeRecognizer):
        def predict(self, *a, **k):
            raise AssertionError("model must not run on invalid input")

    with pytest.raises(ImageValidationError):
        recognize_food(b"junk", Boom([]), db)


def test_manual_correction_overrides_model(db):
    result = recognize_food(png_bytes(), FakeRecognizer([RawPrediction("omelette", 0.9)]), db)
    assert apply_manual_correction(result, ["milk_whole", "milk_whole", "rice_cooked"], db) == ["milk_whole", "rice_cooked"]
    with pytest.raises(KeyError):
        apply_manual_correction(result, ["bogus"], db)
