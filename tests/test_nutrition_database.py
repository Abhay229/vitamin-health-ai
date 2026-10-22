import pandas as pd
import pytest

from src.food_intelligence.nutrition_database import (
    FOOD_DATA_PATH, REQUIRED_COLUMNS, NutritionDataError, NutritionDatabase, validate_dataframe,
)


def raw() -> pd.DataFrame:
    return pd.read_csv(FOOD_DATA_PATH, dtype=str, keep_default_na=False, na_values=[""])


def test_required_columns_exist():
    assert set(REQUIRED_COLUMNS) <= set(raw().columns)


def test_shipped_csv_is_valid_and_ids_unique(db):
    assert validate_dataframe(raw()) == []
    assert raw()["food_id"].is_unique
    assert len(db) >= 20


def test_every_record_has_traceable_source():
    frame = raw()
    for column in ("source_name", "source_reference", "data_version", "data_quality_note"):
        assert frame[column].notna().all() and (frame[column].str.strip() != "").all()


def test_missing_values_are_none_not_zero(db):
    paneer = db.get("paneer")
    assert paneer.b12_ug_per_100g is None and not paneer.has_b12_value
    assert db.get("fortified_cereal").b12_ug_per_100g is None
    zero_plants = [i for i in db.all_items() if i.b12_value_status == "reported_zero_plant"]
    assert zero_plants and all(i.b12_ug_per_100g == 0.0 and i.category == "plant" for i in zero_plants)


def test_fortified_flag_and_category_agree(db):
    fortified = [i for i in db.all_items() if i.is_fortified]
    assert fortified and all(i.category == "fortified" for i in fortified)
    assert not any(i.is_fortified for i in db.all_items() if i.category != "fortified")


def test_required_food_groups_present(db):
    cats = {i.category for i in db.all_items()}
    assert {"dairy", "egg", "fish", "poultry", "red_meat", "fortified", "plant"} <= cats


@pytest.mark.parametrize("mutation, expected", [
    (lambda f: f.__setitem__("b12_ug_per_100g", ["-1"] + list(f["b12_ug_per_100g"][1:])), "negative"),
    (lambda f: f.__setitem__("food_id", ["egg_boiled"] * len(f)), "Duplicate"),
    (lambda f: f.__setitem__("source_reference", [None] + list(f["source_reference"][1:])), "source_reference"),
    (lambda f: f.__setitem__("b12_ug_per_100g", ["abc"] + list(f["b12_ug_per_100g"][1:])), "not numeric"),
])
def test_validation_catches_bad_data(mutation, expected):
    frame = raw()
    mutation(frame)
    assert any(expected in problem for problem in validate_dataframe(frame))


def test_missing_status_with_value_is_rejected():
    frame = raw()
    idx = frame.index[frame["food_id"] == "paneer"][0]
    frame.loc[idx, "b12_ug_per_100g"] = "0"
    assert any("requires an empty" in p for p in validate_dataframe(frame))
    with pytest.raises(NutritionDataError):
        NutritionDatabase(frame)


def test_missing_columns_reported():
    assert "Missing required columns" in validate_dataframe(raw().drop(columns=["source_name"]))[0]


def test_search_and_unknown_id(db):
    assert any(i.food_id == "curd_plain" for i in db.search("dahi"))
    assert db.search("   ") == []
    with pytest.raises(KeyError):
        db.get("nope")


def test_volume_only_for_documented_density(db):
    assert db.get("milk_whole").supports_volume_input
    assert not db.get("egg_boiled").supports_volume_input
