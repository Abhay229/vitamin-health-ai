"""Prediction explanations and scenarios built on the current predictor API."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from src.predict import predict_vitamin_d
from src.ui.constants import GAUGE_MAX


def clamp(value: float, minimum: float, maximum: float) -> float:
    return float(max(minimum, min(maximum, value)))


def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    """Calculate BMI, rejecting an invalid height before prediction."""
    height_m = height_cm / 100.0
    if height_m <= 0:
        raise ValueError("Height must be greater than zero.")
    return float(weight_kg / (height_m * height_m))


def estimate_body_fat_percent(age: float, bmi: float, gender: float) -> float:
    """Estimate body fat with the equations used by the previous deployment."""
    sex = 1.0 if float(gender) >= 0.5 else 0.0
    estimate = (
        1.20 * bmi + 0.23 * age - 10.8 * sex - 5.4
        if age >= 18
        else 1.51 * bmi - 0.70 * age - 3.6 * sex + 1.4
    )
    return clamp(estimate, 5.0, 60.0)


def enrich_model_inputs(inputs: dict) -> dict:
    """Supply the model's derived BMI and body-fat fields without altering other inputs."""
    enriched = dict(inputs)
    bmi = calculate_bmi(float(enriched["Weight_kg"]), float(enriched["Height_cm"]))
    enriched["BMI"] = round(bmi, 2)
    enriched["BodyFat_percent"] = round(
        estimate_body_fat_percent(float(enriched["Age"]), bmi, float(enriched["Gender"])), 2
    )
    return enriched


def explain_prediction(inputs: dict) -> list[str]:
    """Return transparent, deterministic lifestyle signals for the displayed profile."""
    reasons: list[str] = []
    if inputs["Sun_Exposure_min"] < 20:
        reasons.append("Sun exposure is below 20 min/day, which can reduce natural Vitamin D synthesis.")
    if inputs["Indoor_work_hours_day"] > 8:
        reasons.append("Indoor work time is high, so UV exposure opportunities are limited.")
    if inputs["Fish_intake_week"] + inputs["Dairy_intake_week"] < 5:
        reasons.append("Dietary Vitamin D sources look low across fish and dairy intake.")
    if inputs["Physical_activity_hours_week"] < 3:
        reasons.append("Physical activity is on the lower side, which may correlate with less outdoor exposure.")
    if inputs["BMI"] >= 30:
        reasons.append("Higher BMI can be associated with lower circulating Vitamin D levels.")
    if len(reasons) < 2 and inputs["Sun_Exposure_min"] >= 30:
        reasons.append("Current sun exposure is generally supportive for Vitamin D production.")
    if len(reasons) < 2 and inputs["Fish_intake_week"] + inputs["Dairy_intake_week"] >= 7:
        reasons.append("Fish and dairy intake are supportive sources of dietary Vitamin D.")
    return (reasons or ["No major lifestyle risk signal is prominent based on the provided inputs."])[:4]


def build_recommendations(inputs: dict) -> list[str]:
    """Provide the retained offline, non-diagnostic recommendations."""
    recommendations: list[str] = []
    if inputs["Sun_Exposure_min"] < 20:
        recommendations.append("Increase safe daytime sun exposure toward 20-30 minutes on most days.")
    if inputs["Indoor_work_hours_day"] > 8:
        recommendations.append("Add brief daylight breaks during long indoor work periods.")
    if inputs["Fish_intake_week"] + inputs["Dairy_intake_week"] < 5:
        recommendations.append("Add more Vitamin D-rich foods such as fatty fish, eggs, and fortified dairy.")
    if inputs["Physical_activity_hours_week"] < 3:
        recommendations.append("Increase weekly activity to at least 3-5 hours, with some outdoor sessions.")
    baseline = [
        "Track progress for 8-12 weeks and recheck Vitamin D through a lab test.",
        "Discuss supplementation with a clinician if levels remain low despite lifestyle changes.",
    ]
    return (recommendations + baseline)[:4]


def _predict(inputs: dict, model, predictor: Callable[[dict, object], float]) -> float:
    return float(predictor(inputs, model=model))


def build_interpretation_table(
    model,
    base_inputs: dict,
    predictor: Callable[[dict, object], float] = predict_vitamin_d,
) -> pd.DataFrame:
    """Compare current prediction with bounded sunlight and diet scenarios."""
    baseline_inputs = enrich_model_inputs(base_inputs)
    sun_inputs = enrich_model_inputs({
        **baseline_inputs,
        "Sun_Exposure_min": clamp(baseline_inputs["Sun_Exposure_min"] + 20, 0, 180),
        "Skin_Exposure_percent": clamp(baseline_inputs["Skin_Exposure_percent"] + 15, 0, 100),
    })
    diet_inputs = enrich_model_inputs({
        **baseline_inputs,
        "Fish_intake_week": clamp(baseline_inputs["Fish_intake_week"] + 3, 0, 14),
        "Dairy_intake_week": clamp(baseline_inputs["Dairy_intake_week"] + 3, 0, 14),
    })
    baseline_prediction = _predict(baseline_inputs, model, predictor)
    sun_prediction = _predict(sun_inputs, model, predictor)
    diet_prediction = _predict(diet_inputs, model, predictor)
    return pd.DataFrame([
        {"Scenario": "Current baseline", "Feature Changes": "No feature change", "Predicted Vitamin D (ng/mL)": round(baseline_prediction, 2), "Change vs Baseline (ng/mL)": 0.0},
        {"Scenario": "If sun exposure is increased", "Feature Changes": "Sun exposure and skin exposure increased", "Predicted Vitamin D (ng/mL)": round(sun_prediction, 2), "Change vs Baseline (ng/mL)": round(sun_prediction - baseline_prediction, 2)},
        {"Scenario": "If diet intake is increased", "Feature Changes": "Fish and dairy intake increased", "Predicted Vitamin D (ng/mL)": round(diet_prediction, 2), "Change vs Baseline (ng/mL)": round(diet_prediction - baseline_prediction, 2)},
    ])


def build_gauge(prediction: float):
    """Build the retained gauge lazily so non-UI helpers stay dependency-light."""
    import plotly.graph_objects as go

    figure = go.Figure(go.Indicator(
        mode="gauge+number", value=prediction, number={"suffix": " ng/mL"},
        gauge={"axis": {"range": [0, GAUGE_MAX]}, "bar": {"color": "#2563eb"}, "steps": [
            {"range": [0, 20], "color": "#fecaca"},
            {"range": [20, 30], "color": "#fde68a"},
            {"range": [30, GAUGE_MAX], "color": "#bbf7d0"},
        ]},
    ))
    figure.update_layout(height=205, margin={"l": 8, "r": 8, "t": 16, "b": 6})
    return figure
