from datetime import date, timedelta

import pytest

from src.food_intelligence.b12_calculator import compute_intakes, summarize_by_day
from src.food_intelligence.dietary_analysis import (
    REFERENCES, analyze_period, assess_completeness, compare_to_reference, find_patterns, get_reference, screening_indicator,
)
from src.food_intelligence.food_log import make_entry
from src.food_intelligence.recommendations import DISCLAIMER, build_recommendations

REF = get_reference("us_adult")
D = date(2026, 3, 10)


def day_entries(day, foods):
    meals = ["Breakfast", "Lunch", "Dinner", "Snacks"]
    return [make_entry(day, meals[i % 4], fid, grams) for i, (fid, grams) in enumerate(foods)]


def analysis(db, entries, end=D, ref=REF):
    intakes = compute_intakes(entries, db)
    return intakes, analyze_period(intakes, ref, end)


def test_reference_values_are_configurable_and_labelled_as_us():
    assert REFERENCES["us_adult"].value_ug_per_day == 2.4
    assert "US" in REFERENCES["us_adult"].label and "universal" in REFERENCES["us_adult"].note
    assert len({r.value_ug_per_day for r in REFERENCES.values()}) > 3
    with pytest.raises(KeyError):
        get_reference("x")


def test_comparison_levels_and_numbers(db):
    full = day_entries(D, [("egg_boiled", 100), ("milk_whole", 200), ("sardine_canned_oil", 85)])  # 1.11+0.9+7.6=9.6
    _, result = analysis(db, full)
    assert result.today.level == "at_or_above" and result.today.difference_ug > 0
    assert result.today.percent_of_reference == pytest.approx(result.today.estimated_ug / 2.4 * 100)
    low = day_entries(D, [("milk_whole", 100)])  # 0.45
    assert analysis(db, low)[1].today.level == "below"
    near = day_entries(D, [("egg_boiled", 100), ("milk_whole", 100), ("chicken_breast_roasted", 100)])  # 1.9 -> 79 %
    assert analysis(db, near)[1].today.level == "near"


def test_empty_log_is_insufficient_not_zero(db):
    _, result = analysis(db, [])
    assert result.today.level == "insufficient_data" and result.today.estimated_ug is None
    assert result.window.estimated_ug is None and result.completeness.level == "none"


def test_missing_days_not_treated_as_zero(db):
    entries = day_entries(D, [("egg_boiled", 200), ("milk_whole", 250)]) + day_entries(D - timedelta(days=3), [("egg_boiled", 200), ("milk_whole", 250)])
    _, result = analysis(db, entries)
    assert result.window_average.recorded_days == 2
    assert result.window.estimated_ug == pytest.approx(2 * 1.11 + 2.5 * 0.45)  # equals one day's total, not halved/ over 7 days


def test_incomplete_logs_flagged(db):
    one_day = day_entries(D, [("egg_boiled", 50)])
    intakes, result = analysis(db, one_day)
    assert result.completeness.level == "limited"
    assert any("Only 1 day" in r for r in result.completeness.reasons) and any("meal types" in r for r in result.completeness.reasons)
    unknown = day_entries(D, [("paneer", 50)])
    assert any("unknown" in r.lower() for r in analysis(db, unknown)[1].completeness.reasons)


def test_adequate_completeness_requires_days_meals_and_known_values(db):
    entries = []
    for n in range(3):
        entries += day_entries(D - timedelta(days=n), [("egg_boiled", 50), ("milk_whole", 244), ("rice_cooked", 158)])
    assert analysis(db, entries)[1].completeness.is_sufficient


def test_patterns_have_evidence_and_come_from_log(db):
    entries = []
    for n in range(3):
        entries += day_entries(D - timedelta(days=n), [("rice_cooked", 158), ("lentils_cooked", 198), ("banana_raw", 118)])
    intakes = compute_intakes(entries, db)
    findings = {f.rule_id: f for f in find_patterns(intakes, REF, D, "non_vegetarian")}
    assert {"no_b12_foods", "limited_animal_source", "limited_fortified"} <= set(findings)
    assert all(f.evidence for f in findings.values())
    assert "may be low" in findings["no_b12_foods"].explanation and "cannot determine" in findings["no_b12_foods"].explanation


def test_repeated_low_requires_multiple_days(db):
    entries = []
    for n in range(3):
        entries += day_entries(D - timedelta(days=n), [("milk_whole", 244), ("rice_cooked", 158), ("banana_raw", 118)])
    ids = {f.rule_id for f in find_patterns(compute_intakes(entries, db), REF, D, "vegetarian")}
    assert "repeated_low" in ids
    assert "repeated_low" not in {f.rule_id for f in find_patterns(compute_intakes(entries[:3], db), REF, D, "vegetarian")}


def test_vegan_preference_adapts_and_is_not_assumed_deficient(db):
    intakes = compute_intakes(day_entries(D, [("rice_cooked", 158), ("lentils_cooked", 198), ("banana_raw", 118)]), db)
    ids = {f.rule_id for f in find_patterns(intakes, REF, D, "vegan")}
    assert "no_fortified_vegan" in ids and "limited_animal_source" not in ids
    result = analyze_period(intakes, REF, D)
    tips = " ".join(build_recommendations(result, find_patterns(intakes, REF, D, "vegan"), "vegan")).lower()
    assert "fortified" in tips and "healthcare professional" in tips and "egg" not in tips


def test_fortified_label_value_counts_for_vegan(db):
    entries = day_entries(D, [("fortified_plant_milk", 250)])
    entries[0] = make_entry(D, "Breakfast", "fortified_plant_milk", 250, user_b12_ug_per_100g=0.4)
    intakes = compute_intakes(entries, db)
    assert "no_fortified_vegan" not in {f.rule_id for f in find_patterns(intakes, REF, D, "vegan")}


def test_indicator_is_never_a_diagnosis(db):
    for foods in ([("milk_whole", 100)], [("egg_boiled", 300), ("sardine_canned_oil", 85)], []):
        intakes, result = analysis(db, day_entries(D, foods))
        ind = screening_indicator(result, find_patterns(intakes, REF, D))
        text = (ind.label + " " + ind.explanation).lower()
        assert "not a diagnosis" in text and "screening indicator" in text
        assert "deficien" not in ind.label.lower() and "pg/ml" not in text


def test_no_fabricated_blood_levels_or_doses_in_recommendations(db):
    intakes, result = analysis(db, day_entries(D, [("milk_whole", 100)]))
    text = " ".join(build_recommendations(result, find_patterns(intakes, REF, D), "non_vegetarian") + [DISCLAIMER])
    assert "pg/mL" not in text and " mcg" not in text and "1000" not in text and "cure" not in text.lower()
    assert "not a substitute for medical advice" in DISCLAIMER


def test_completeness_none_for_empty():
    assert assess_completeness({}, []).level == "none"


def test_compare_rejects_bad_value():
    with pytest.raises(ValueError):
        compare_to_reference(-1.0, REF, assess_completeness({}, []))
