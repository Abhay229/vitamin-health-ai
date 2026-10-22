import pytest

from src.ui.insights import (
    build_interpretation_table,
    calculate_bmi,
    enrich_model_inputs,
)


def sample_inputs():
    return {
        "Age": 25, "Gender": 1, "Weight_kg": 70.0, "Height_cm": 175.0,
        "Sun_Exposure_min": 10.0, "Skin_Exposure_percent": 30.0,
        "Sunscreen_Use": 0, "Alcohol_units_week": 0.0, "Fish_intake_week": 1.0,
        "Dairy_intake_week": 2.0, "Physical_activity_hours_week": 2.0,
        "Indoor_work_hours_day": 9.0, "Skin_tone": 2,
    }


def test_enrichment_calculates_model_derived_inputs():
    enriched = enrich_model_inputs(sample_inputs())
    assert enriched["BMI"] == pytest.approx(22.86)
    assert 5 <= enriched["BodyFat_percent"] <= 60


def test_bmi_rejects_zero_height():
    with pytest.raises(ValueError, match="Height"):
        calculate_bmi(70, 0)


def test_what_if_scenarios_use_current_predictor_interface():
    calls = []

    def fake_predictor(inputs, model=None):
        calls.append(inputs.copy())
        return inputs["Sun_Exposure_min"] + inputs["Fish_intake_week"]

    table = build_interpretation_table(object(), sample_inputs(), predictor=fake_predictor)

    assert len(calls) == 3
    assert calls[1]["Sun_Exposure_min"] == 30
    assert calls[2]["Fish_intake_week"] == 4
    assert table["Scenario"].tolist() == [
        "Current baseline", "If sun exposure is increased", "If diet intake is increased",
    ]
