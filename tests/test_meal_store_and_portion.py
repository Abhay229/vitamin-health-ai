from datetime import date

from evaluation.intake_error import intake_error, recognition_metrics
from src.food_intelligence import meal_store
from src.food_intelligence.food_log import FoodLog, make_entry
from src.food_intelligence.nutrition_database import NutritionDatabase
from src.food_intelligence.portion_estimator import grams_from_ml, grams_from_servings


def test_sqlite_roundtrip_is_idempotent(tmp_path):
    path = tmp_path / "m.db"
    log = FoodLog([make_entry(date(2025, 1, 1), "Lunch", "x", 100.0, entry_id="a")])
    meal_store.save_log(log, path)
    meal_store.save_log(log, path)
    restored = FoodLog.from_records(meal_store.load_records(path))
    assert len(restored) == 1 and restored.get("a").portion_g == 100.0


def test_portion_conversions_need_documented_basis():
    items = NutritionDatabase.from_csv().all_items()
    serving = next(i for i in items if i.has_standard_serving)
    assert grams_from_servings(serving, 2) == serving.serving_weight_g * 2
    no_density = next(i for i in items if not i.supports_volume_input)
    assert grams_from_ml(no_density, 250) is None


def test_metrics():
    assert intake_error([2, 4], [1, 4])["mae_ug"] == 0.5
    assert recognition_metrics(["a", "b"], ["a", "c"])["correction_rate"] == 0.5
