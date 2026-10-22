"""Polished Streamlit interface backed by the current saved Vitamin D pipeline."""

from __future__ import annotations

import streamlit as st

from src.predict import load_metadata, load_model, predict_vitamin_d, vitamin_d_status
from src.ui.chatbot import render_chat_widget
from src.ui.food_page import render_food_page
from src.ui.history_page import render_history_page
from src.ui.constants import DETAILED_MODE, GAUGE_MAX, LAYOUT, PAGE_ICON, PAGE_TITLE, QUICK_MODE
from src.ui.insights import build_gauge, build_interpretation_table, build_recommendations, enrich_model_inputs, explain_prediction
from src.ui.styles import apply_global_styles


st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=LAYOUT)
apply_global_styles()


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def render_input_panel() -> tuple[bool, dict, str]:
    """Render the former app's two-mode experience with the current model schema."""
    defaults = st.session_state.get("last_model_input", {})
    default_mode = st.session_state.get("last_mode", QUICK_MODE)
    gender_labels = {"Female": 0, "Male": 1}
    default_gender = "Male" if float(defaults.get("Gender", 1)) >= 0.5 else "Female"

    with st.container():
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="mode-switch-label">Choose Analysis Mode</div>', unsafe_allow_html=True)
        mode = st.radio("Input Mode", [QUICK_MODE, DETAILED_MODE], horizontal=True, index=0 if default_mode == QUICK_MODE else 1, label_visibility="collapsed")
        st.markdown('<div class="mode-label">Fast screening using core inputs only.</div>' if mode == QUICK_MODE else '<div class="mode-label">Richer profile inputs for context and recommendations.</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Personal Info</div>', unsafe_allow_html=True)
        first, second = st.columns(2)
        with first:
            age = st.number_input("Age", 1, 100, int(clamp(float(defaults.get("Age", 25)), 1, 100)))
            height = st.number_input("Height (cm)", 100.0, 230.0, float(clamp(float(defaults.get("Height_cm", 170)), 100, 230)), step=0.5)
        with second:
            gender = st.selectbox("Gender", list(gender_labels), index=1 if default_gender == "Male" else 0)
            weight = st.number_input("Weight (kg)", 25.0, 250.0, float(clamp(float(defaults.get("Weight_kg", 70)), 25, 250)), step=0.5)

        preview = enrich_model_inputs({"Age": age, "Gender": gender_labels[gender], "Weight_kg": weight, "Height_cm": height})
        metric_one, metric_two = st.columns(2)
        metric_one.metric("Calculated BMI", f"{preview['BMI']:.2f}")
        metric_two.metric("Estimated Body Fat %", f"{preview['BodyFat_percent']:.2f}%")
        st.caption("BMI and body-fat estimate are calculated automatically from the profile fields.")
        st.markdown('<div class="section-title">Lifestyle</div>', unsafe_allow_html=True)
        first, second = st.columns(2)
        with first:
            sun = st.slider("Sun Exposure (min/day)", 0, 180, int(clamp(float(defaults.get("Sun_Exposure_min", 30)), 0, 180)), 5)
        with second:
            activity = st.slider("Physical Activity (hours/week)", 0.0, 20.0, float(clamp(float(defaults.get("Physical_activity_hours_week", 3)), 0, 20)), 0.5)

        skin_exposure, sunscreen, indoor, fish, dairy, alcohol, skin_tone = 40, 0, 7.0, 2, 4, 0, 2
        if mode == DETAILED_MODE:
            st.markdown('<div class="section-title">Detailed Profile</div>', unsafe_allow_html=True)
            first, second = st.columns(2)
            with first:
                skin_exposure = st.slider("Skin Exposure (%)", 0, 100, int(clamp(float(defaults.get("Skin_Exposure_percent", 40)), 0, 100)), 5)
                sunscreen = st.selectbox("Sunscreen use", [0, 1], index=int(clamp(float(defaults.get("Sunscreen_Use", 0)), 0, 1)), help="Uses the dataset's original 0/1 coding.")
                fish = st.slider("Fish intake/week", 0, 14, int(clamp(float(defaults.get("Fish_intake_week", 2)), 0, 14)))
                alcohol = st.slider("Alcohol units/week", 0, 30, int(clamp(float(defaults.get("Alcohol_units_week", 0)), 0, 30)))
            with second:
                indoor = st.slider("Indoor work hours/day", 0.0, 16.0, float(clamp(float(defaults.get("Indoor_work_hours_day", 7)), 0, 16)), 0.5)
                skin_tone = st.selectbox("Skin tone coding", [1, 2, 3], index=int(clamp(float(defaults.get("Skin_tone", 2)), 1, 3)) - 1, help="Uses the dataset's original coding.")
                dairy = st.slider("Dairy intake/week", 0, 14, int(clamp(float(defaults.get("Dairy_intake_week", 4)), 0, 14)))
        predict_clicked = st.button("Predict Vitamin D", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    inputs = enrich_model_inputs({
        "Age": age, "Gender": gender_labels[gender], "Weight_kg": weight, "Height_cm": height,
        "Sun_Exposure_min": sun, "Skin_Exposure_percent": skin_exposure, "Sunscreen_Use": sunscreen,
        "Alcohol_units_week": alcohol, "Fish_intake_week": fish, "Dairy_intake_week": dairy,
        "Physical_activity_hours_week": activity, "Indoor_work_hours_day": indoor, "Skin_tone": skin_tone,
    })
    return predict_clicked, inputs, mode


def render_output_panel(model, metadata: dict, model_input: dict, active_mode: str) -> None:
    """Display results while keeping the current saved pipeline authoritative."""
    prediction = predict_vitamin_d(model_input, model=model)
    status = vitamin_d_status(prediction)
    status_class = {"Deficient": "status-deficient", "Insufficient": "status-insufficient", "Sufficient": "status-sufficient"}[status]
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Prediction Output")
    summary, gauge = st.columns([2, 1], gap="medium")
    with summary:
        st.markdown(f'<div class="result-value">{prediction:.2f} ng/mL</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="status-chip {status_class}">Status: {status}</div>', unsafe_allow_html=True)
        st.caption(f"Calculated BMI: {model_input['BMI']:.2f} | Estimated body fat: {model_input['BodyFat_percent']:.2f}%")
        st.progress(min(max(prediction / GAUGE_MAX, 0.0), 1.0))
        st.caption(metadata["data_label"])
    with gauge:
        st.plotly_chart(build_gauge(prediction), use_container_width=True, config={"displayModeBar": False})
    if active_mode == DETAILED_MODE:
        st.divider()
        st.subheader("Why this?")
        st.markdown("\n".join(f"- {reason}" for reason in explain_prediction(model_input)))
        st.divider()
        st.subheader("Recommendations")
        st.markdown("\n".join(f"- {tip}" for tip in build_recommendations(model_input)))
        st.divider()
        st.subheader("What-If Analysis")
        table = build_interpretation_table(model, model_input)
        st.dataframe(table, use_container_width=True, hide_index=True)
        best = table.iloc[1:].sort_values("Predicted Vitamin D (ng/mL)", ascending=False).iloc[0]
        st.success(f"Most impactful scenario: {best['Scenario']} ({best['Change vs Baseline (ng/mL)']:+.2f} ng/mL vs baseline).")
        render_chat_widget(model_input, prediction, status)
    st.markdown("</div>", unsafe_allow_html=True)


PAGE_VITAMIN_D = "Vitamin D Analysis"
PAGE_FOOD = "Food Intelligence — Vitamin B12"
PAGE_HISTORY = "Nutrition History"
PAGES = (PAGE_VITAMIN_D, PAGE_FOOD, PAGE_HISTORY)


def render_vitamin_d_page() -> None:
    """The original Vitamin D workflow, unchanged apart from being wrapped in a function."""
    st.markdown(f'<div class="hero-card"><div class="hero-eyebrow">Vitamin D Wellness Forecast</div><h1>{PAGE_TITLE}</h1><p>Estimate Vitamin D from everyday health and lifestyle inputs in a guided experience.</p></div>', unsafe_allow_html=True)
    try:
        metadata = load_metadata()
        model = load_model()
    except FileNotFoundError:
        st.error("Model files are missing. Run `python -m src.train` from the project folder first.")
        st.stop()

    has_prediction = "last_model_input" in st.session_state
    if has_prediction:
        with st.sidebar:
            st.subheader("Input Form")
            st.caption("Update values and select Predict Vitamin D to refresh results.")
            predict_clicked, model_input, mode = render_input_panel()
    else:
        left, right = st.columns([1, 1], gap="medium")
        with left:
            predict_clicked, model_input, mode = render_input_panel()
        with right:
            if "last_model_input" in st.session_state:
                render_output_panel(model, metadata, st.session_state["last_model_input"], st.session_state.get("last_mode", QUICK_MODE))
    if predict_clicked:
        st.session_state["last_model_input"] = model_input
        st.session_state["last_mode"] = mode
        st.rerun()
    if "last_model_input" in st.session_state and has_prediction:
        render_output_panel(model, metadata, st.session_state["last_model_input"], st.session_state.get("last_mode", QUICK_MODE))


def render_navigation() -> str:
    """Top-level navigation between the Vitamin D module and the B12 food module."""
    if st.session_state.get("active_page") not in PAGES:
        st.session_state["active_page"] = PAGE_VITAMIN_D
    return st.radio("Navigate", PAGES, horizontal=True, key="active_page", label_visibility="collapsed")


page = render_navigation()
if page == PAGE_FOOD:
    render_food_page()
elif page == PAGE_HISTORY:
    render_history_page()
else:
    render_vitamin_d_page()
