from datetime import date, timedelta

import pytest

from src.food_intelligence.b12_calculator import (
    InvalidNutrientValueError, InvalidPortionError, calculate_b12_ug, calculate_entry_b12, compute_intakes, entry_intake,
    filter_window, format_ug, summarize, summarize_by_day, summarize_by_meal, volume_ml_to_grams, window_average,
)
from src.food_intelligence.food_log import FoodLog, make_entry


def test_formula_known_values():
    assert calculate_b12_ug(200, 1.3) == pytest.approx(2.6)  # arithmetic example from the spec
    assert calculate_b12_ug(100, 0.0) == 0.0


def test_entry_matches_database_record(db, today):
    item = db.get("egg_boiled")
    e = make_entry(today, "Breakfast", "egg_boiled", 100)
    assert entry_intake(e, db).b12_ug == pytest.approx(item.b12_ug_per_100g)  # 100 g -> exactly the per-100 g value


@pytest.mark.parametrize("bad", [0, -5, float("nan"), float("inf"), "x", None, 5001])
def test_invalid_portions_rejected(bad):
    with pytest.raises(InvalidPortionError):
        calculate_b12_ug(bad, 1.0)
    with pytest.raises(InvalidPortionError):
        calculate_entry_b12(bad, None)  # invalid portion is not hidden by missing data


@pytest.mark.parametrize("bad", [-0.1, float("nan"), float("inf"), "x"])
def test_invalid_nutrient_values_rejected(bad):
    with pytest.raises(InvalidNutrientValueError):
        calculate_b12_ug(100, bad)


def test_missing_value_is_unknown_not_zero(db, today):
    row = entry_intake(make_entry(today, "Lunch", "paneer", 100), db)
    assert row.b12_ug is None and row.b12_basis == "missing"
    summary = summarize([row])
    assert summary.known_total_ug == 0.0 and summary.entries_unknown == 1 and summary.is_lower_bound and not summary.is_complete
    assert format_ug(None) == "unknown"


def test_user_label_fills_missing_but_never_overrides_database(db, today):
    filled = entry_intake(make_entry(today, "Breakfast", "fortified_cereal", 40, user_b12_ug_per_100g=3.0), db)
    assert filled.b12_ug == pytest.approx(1.2) and filled.b12_basis == "user_label" and filled.is_fortified
    ignored = entry_intake(make_entry(today, "Breakfast", "egg_boiled", 100, user_b12_ug_per_100g=99), db)
    assert ignored.b12_basis == "database" and ignored.b12_ug == pytest.approx(db.get("egg_boiled").b12_ug_per_100g)


def test_meal_and_day_aggregation_and_shares(db, today):
    entries = [make_entry(today, "Breakfast", "egg_boiled", 100), make_entry(today, "Breakfast", "milk_whole", 200),
               make_entry(today, "Lunch", "rice_cooked", 150), make_entry(today, "Dinner", "fortified_cereal", 30, user_b12_ug_per_100g=4.0)]
    intakes = compute_intakes(entries, db)
    day = summarize(intakes)
    expected = 1.11 + 0.9 + 0.0 + 1.2
    assert day.known_total_ug == pytest.approx(expected)
    assert sum(c.share_of_known for c in day.contributions) == pytest.approx(1.0)
    assert day.fortified_ug == pytest.approx(1.2) and day.natural_ug == pytest.approx(2.01)
    meals = summarize_by_meal(intakes)
    assert meals["Breakfast"].known_total_ug == pytest.approx(2.01)
    assert summarize_by_day(intakes)[today].known_total_ug == pytest.approx(expected)


def test_empty_log_summaries():
    s = summarize([])
    assert s.is_empty and s.known_total_ug == 0.0 and not s.is_complete and s.contributions == ()
    assert summarize_by_day([]) == {}
    assert window_average({}, date(2026, 3, 10)).average_ug is None


def test_weekly_average_uses_recorded_days_only(db, today):
    entries = [make_entry(today, "Lunch", "egg_boiled", 100), make_entry(today - timedelta(days=2), "Lunch", "egg_boiled", 300)]
    avg = window_average(summarize_by_day(compute_intakes(entries, db)), today, 7)
    assert avg.recorded_days == 2 and avg.average_ug == pytest.approx((1.11 + 3.33) / 2)  # 5 empty days are NOT zeros


def test_window_excludes_old_days(db, today):
    entries = [make_entry(today - timedelta(days=10), "Lunch", "egg_boiled", 100)]
    intakes = compute_intakes(entries, db)
    assert filter_window(intakes, today, 7) == []
    assert window_average(summarize_by_day(intakes), today, 7).average_ug is None


def test_volume_conversion_requires_documented_density(db):
    assert volume_ml_to_grams(200, db.get("milk_whole").density_g_per_ml) == pytest.approx(206.0)
    with pytest.raises(InvalidPortionError):
        volume_ml_to_grams(200, None)
    with pytest.raises(InvalidPortionError):
        volume_ml_to_grams(-1, 1.0)


def test_edit_does_not_double_count(db, today):
    log = FoodLog([make_entry(today, "Lunch", "egg_boiled", 100, entry_id="a")])
    before = summarize(compute_intakes(log.entries(), db)).known_total_ug
    log.update("a", portion_g=200)
    after = summarize(compute_intakes(log.entries(), db)).known_total_ug
    assert len(log) == 1 and after == pytest.approx(before * 2)
    log.update("a", food_id="milk_whole", portion_g=100)
    assert summarize(compute_intakes(log.entries(), db)).known_total_ug == pytest.approx(0.45)


def test_precision_retained_internally(db, today):
    value = entry_intake(make_entry(today, "Lunch", "egg_boiled", 33.3333), db).b12_ug
    assert value == pytest.approx(33.3333 * 1.11 / 100, rel=1e-12) and format_ug(value) == "0.37 µg"
