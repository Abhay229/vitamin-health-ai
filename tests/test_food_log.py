from datetime import date, timedelta

import pytest

from src.food_intelligence.food_log import FoodLog, FoodLogError, MEALS, get_session_log, make_entry


D = date(2026, 3, 10)


def test_add_is_idempotent_for_reruns():
    log = FoodLog()
    entry = make_entry(D, "Breakfast", "egg_boiled", 50, entry_id="draft1")
    assert log.add(entry) is True
    assert log.add(entry) is False  # same draft re-submitted after a Streamlit rerun
    assert len(log) == 1


def test_update_keeps_single_entry_and_validates():
    log = FoodLog([make_entry(D, "Lunch", "egg_boiled", 50, entry_id="a")])
    log.update("a", portion_g=75, meal="Dinner")
    assert len(log) == 1 and log.get("a").portion_g == 75 and log.get("a").meal == "Dinner"
    with pytest.raises(FoodLogError):
        log.update("a", portion_g=-1)
    assert log.get("a").portion_g == 75  # failed update changed nothing
    with pytest.raises(FoodLogError):
        log.update("a", entry_id="b")
    with pytest.raises(FoodLogError):
        log.update("zzz", portion_g=1)


def test_remove_and_clear_day():
    log = FoodLog([make_entry(D, "Lunch", "egg_boiled", 50, entry_id="a"), make_entry(D, "Dinner", "milk_whole", 200, entry_id="b"),
                   make_entry(D - timedelta(days=1), "Lunch", "egg_boiled", 50, entry_id="c")])
    assert log.remove("a") and not log.remove("a")
    assert log.clear_day(D) == 1 and len(log) == 1 and log.dates() == [D - timedelta(days=1)]


@pytest.mark.parametrize("kwargs", [
    dict(meal="Brunch"), dict(portion_g=0), dict(portion_g=-3), dict(portion_g=float("nan")), dict(portion_g=10**6),
    dict(source="telepathy"), dict(user_b12_ug_per_100g=-1), dict(food_id=""),
])
def test_invalid_entries_rejected(kwargs):
    base = dict(log_date=D, meal="Lunch", food_id="egg_boiled", portion_g=50)
    base.update(kwargs)
    with pytest.raises(FoodLogError):
        make_entry(**base)


def test_filtering_and_meals_constant():
    log = FoodLog([make_entry(D, "Lunch", "egg_boiled", 50), make_entry(D, "Snacks", "almonds_raw", 28)])
    assert MEALS == ("Breakfast", "Lunch", "Dinner", "Snacks")
    assert len(log.entries(D, "Snacks")) == 1 and len(log.entries(D)) == 2


def test_record_roundtrip_and_merge_skips_duplicates():
    log = FoodLog([make_entry(D, "Lunch", "fortified_cereal", 30, user_b12_ug_per_100g=2.5, entry_id="a")])
    restored = FoodLog.from_records(log.to_records())
    assert restored.get("a") == log.get("a")
    assert restored.merge_records(log.to_records()) == (0, 1)
    with pytest.raises(FoodLogError):
        FoodLog.from_records([{"entry_id": "x", "log_date": "not-a-date", "meal": "Lunch", "food_id": "a", "portion_g": 1}])


def test_session_log_is_created_once():
    state = {}
    assert get_session_log(state) is get_session_log(state)
