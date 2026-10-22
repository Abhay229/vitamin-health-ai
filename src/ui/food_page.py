"""Streamlit page: Food Intelligence — Vitamin B12 intake estimation."""

from __future__ import annotations

import hashlib
from datetime import date

import pandas as pd
import streamlit as st

from src.food_intelligence import dashboard as charts
from src.food_intelligence.b12_calculator import (
    InvalidPortionError, MAX_PORTION_G, calculate_entry_b12, compute_intakes, filter_window, format_ug,
    summarize, summarize_by_day, summarize_by_meal, volume_ml_to_grams,
)
from src.food_intelligence.dietary_analysis import (
    DEFAULT_REFERENCE_KEY, INDICATOR_NAME, PREFERENCES, REFERENCES, analyze_period, find_patterns, get_reference, screening_indicator,
)
from src.food_intelligence.food_log import MEALS, FoodLog, FoodLogError, get_session_log, make_entry, new_entry_id
from src.food_intelligence.food_recognition import (
    ImageValidationError, RecognitionResult, get_default_recognizer, recognize_food,
)
from src.food_intelligence.nutrition_database import FoodItem, NutritionDatabase
from src.food_intelligence.recommendations import DISCLAIMER, RISK_FACTOR_EDUCATION, build_recommendations

WINDOW_DAYS = 7
PLOT_CONFIG = {"displayModeBar": False}


@st.cache_resource(show_spinner=False)
def load_database() -> NutritionDatabase:
    return NutritionDatabase.from_csv()


def _bump_form() -> None:
    """Reset add-food widgets and mint fresh draft ids (prevents re-adding on reruns)."""
    st.session_state["fi_nonce"] = st.session_state.get("fi_nonce", 0) + 1


def _parse_label_value(text: str) -> tuple[float | None, str | None]:
    text = (text or "").strip()
    if not text:
        return None, None
    try:
        value = float(text)
    except ValueError:
        return None, "Label value must be a number (µg per 100 g)."
    if value < 0 or value != value or value == float("inf"):
        return None, "Label value must not be negative."
    return value, None


def render_hero() -> None:
    st.markdown(
        '<div class="hero-card"><div class="hero-eyebrow">Food Intelligence</div><h1>Vitamin B12 Intake Estimation</h1>'
        "<p>Log meals from a photo or manually, confirm portions, and see your estimated dietary B12 intake and patterns.</p></div>",
        unsafe_allow_html=True)
    st.markdown(f'<div class="notice">{DISCLAIMER}</div>', unsafe_allow_html=True)


def render_settings() -> tuple[date, str, str]:
    with st.expander("Date, dietary preference and reference", expanded=False):
        c1, c2, c3 = st.columns(3)
        log_date = c1.date_input("Log date", value=date.today(), max_value=date.today(), key="fi_date")
        preference = c2.selectbox("Dietary preference", list(PREFERENCES), format_func=PREFERENCES.get, key="fi_pref")
        reference_key = c3.selectbox("Dietary reference", list(REFERENCES), format_func=lambda k: REFERENCES[k].label,
                                     index=list(REFERENCES).index(DEFAULT_REFERENCE_KEY), key="fi_ref")
        ref = get_reference(reference_key)
        st.caption(f"Source: {ref.source}. {ref.note} Session data is not stored on any server by this app.")
    return log_date, preference, reference_key


def _portion_inputs(item: FoodItem, key: str) -> tuple[float | None, str | None]:
    """Return (portion in grams, error). Offers standard serving, custom grams, or volume when documented."""
    modes = (["Standard serving"] if item.has_standard_serving else []) + ["Custom weight (g)"] + (["Volume (ml)"] if item.supports_volume_input else [])
    mode = st.radio("Portion entry", modes, horizontal=True, key=f"{key}_mode")
    try:
        if mode == "Standard serving":
            servings = st.number_input("Number of servings", 0.25, 20.0, 1.0, 0.25, key=f"{key}_servings")
            st.caption(f"1 serving = {item.serving_description} (documented in the nutrition dataset).")
            return float(servings) * item.serving_weight_g, None
        if mode == "Volume (ml)":
            ml = st.number_input("Volume (ml)", 1.0, 5000.0, 200.0, 10.0, key=f"{key}_ml")
            st.caption(f"Converted with the documented density {item.density_g_per_ml:g} g/ml for this food.")
            return volume_ml_to_grams(ml, item.density_g_per_ml), None
        grams = st.number_input("Weight (g)", 1.0, MAX_PORTION_G, float(item.serving_weight_g or 100.0), 5.0, key=f"{key}_g")
        return float(grams), None
    except InvalidPortionError as error:
        return None, str(error)


def render_portion_form(db: NutritionDatabase, food_id: str, source: str, key: str, meal: str, log_date: date, log: FoodLog) -> None:
    """Confirm portion, preview the calculation, then add to the log."""
    item = db.get(food_id)
    portion_g, error = _portion_inputs(item, key)
    label_text = ""
    if not item.has_b12_value:
        label_text = st.text_input("B12 per 100 g from the product label (µg) — optional", key=f"{key}_label",
                                   help="Leave blank if unknown. A blank value keeps this item's B12 UNKNOWN; it is never treated as 0.")
    label_value, label_error = _parse_label_value(label_text)
    error = error or label_error

    density = item.b12_ug_per_100g if item.has_b12_value else label_value
    basis = "database" if item.has_b12_value else ("your label value" if label_value is not None else "no value available")
    estimate = None
    if error is None and portion_g is not None:
        estimate = calculate_entry_b12(portion_g, density)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Food", item.food_name)
    c2.metric("Portion", "—" if portion_g is None else f"{portion_g:.0f} g")
    c3.metric("B12 per 100 g", "unknown" if density is None else f"{density:g} µg", help=f"Basis: {basis}")
    c4.metric("Estimated B12", format_ug(estimate))
    st.caption(f"Source: {item.source_name} — {item.source_reference}. {item.data_quality_note}")
    if error:
        st.error(error)
    if density is None:
        st.warning("B12 content is unknown for this food. It can still be logged, but it will be counted as unknown, not as zero.")

    draft_key = f"fi_draft_{key}"
    draft_id = st.session_state.setdefault(draft_key, new_entry_id())
    if st.button(f"Add to {meal}", type="primary", key=f"{key}_add", disabled=error is not None or portion_g is None, use_container_width=True):
        try:
            added = log.add(make_entry(log_date, meal, food_id, portion_g, source=source, user_b12_ug_per_100g=label_value, entry_id=draft_id))
        except FoodLogError as problem:
            st.error(str(problem))
            return
        st.session_state.pop(draft_key, None)
        st.session_state["fi_flash"] = f"Added {item.food_name} to {meal}." if added else "This entry was already added."
        _bump_form()
        st.rerun()


def _food_options(db: NutritionDatabase, preferred: list[str] | None = None) -> list[str]:
    preferred = [p for p in (preferred or []) if p in db]
    return preferred + [i.food_id for i in db.all_items() if i.food_id not in preferred]


def _render_photo_tab(db: NutritionDatabase, meal: str, log_date: date, log: FoodLog, nonce: int) -> None:
    upload = st.file_uploader("Upload a meal photo (JPG, PNG or WEBP, up to 5 MB)", type=["jpg", "jpeg", "png", "webp"], key=f"fi_upload_{nonce}")
    st.caption("Photo recognition only *suggests* foods. It cannot judge portion size, ingredients or fortification. You confirm everything.")
    if upload is None:
        return
    data = upload.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    cached = st.session_state.get("fi_reco")
    if not cached or cached[0] != digest:
        try:
            with st.spinner("Analysing image..."):
                result: RecognitionResult | str = recognize_food(data, get_default_recognizer(), db)
        except ImageValidationError as error:
            result = str(error)
        st.session_state["fi_reco"] = (digest, result)
    result = st.session_state["fi_reco"][1]
    if isinstance(result, str):
        st.error(result + " You can still select the food manually in the next tab.")
        return
    st.image(data, caption="Uploaded image — please check this is the right meal", width=320)
    (st.warning if (result.low_confidence or not result.available) else st.info)(result.message)
    if result.predictions:
        st.dataframe(pd.DataFrame([{"Model label": p.label, "Confidence": f"{p.confidence:.0%}",
                                    "Mapped food": db.get(p.food_id).food_name if p.food_id else "— none —",
                                    "Note": p.note} for p in result.predictions]), hide_index=True, use_container_width=True)
        st.caption(f"Model: {result.model_name}. Accuracy has not been evaluated for this project.")
    chosen = st.multiselect("Confirm the food(s) on the plate (suggestions are listed first; you can add others)",
                            _food_options(db, result.mapped_food_ids), default=[], format_func=lambda f: db.get(f).food_name,
                            key=f"fi_confirm_{nonce}", placeholder="Choose the foods you actually ate")
    for food_id in chosen:
        with st.container(border=True):
            st.markdown(f"**{db.get(food_id).food_name}**")
            render_portion_form(db, food_id, "image", f"img_{nonce}_{food_id}", meal, log_date, log)


def _render_manual_tab(db: NutritionDatabase, meal: str, log_date: date, log: FoodLog, nonce: int) -> None:
    food_id = st.selectbox("Search and select a food", _food_options(db), index=None, placeholder="Type to search, e.g. egg, dahi, fish",
                           format_func=lambda f: db.get(f).food_name, key=f"fi_manual_{nonce}")
    if food_id:
        render_portion_form(db, food_id, "manual", f"man_{nonce}_{food_id}", meal, log_date, log)


def _render_entry_editor(db: NutritionDatabase, entry, log: FoodLog) -> None:
    key = f"edit_{entry.entry_id}"
    options = _food_options(db)
    c1, c2, c3 = st.columns(3)
    food_id = c1.selectbox("Food", options, index=options.index(entry.food_id), format_func=lambda f: db.get(f).food_name, key=f"{key}_food")
    portion = c2.number_input("Portion (g)", 1.0, MAX_PORTION_G, float(entry.portion_g), 5.0, key=f"{key}_portion")
    meal = c3.selectbox("Meal", MEALS, index=MEALS.index(entry.meal), key=f"{key}_meal")
    label_text = ""
    if not db.get(food_id).has_b12_value:
        label_text = st.text_input("B12 per 100 g from label (µg), optional", "" if entry.user_b12_ug_per_100g is None else str(entry.user_b12_ug_per_100g), key=f"{key}_label")
    value, label_error = _parse_label_value(label_text)
    if label_error:
        st.error(label_error)
    b1, b2 = st.columns(2)
    if b1.button("Save changes", key=f"{key}_save", disabled=label_error is not None):
        try:
            log.update(entry.entry_id, food_id=food_id, portion_g=portion, meal=meal, user_b12_ug_per_100g=value)
        except FoodLogError as problem:
            st.error(str(problem))
        else:
            st.rerun()
    if b2.button("Remove entry", key=f"{key}_remove"):
        log.remove(entry.entry_id)
        st.rerun()


def _render_day_log(db: NutritionDatabase, log: FoodLog, log_date: date) -> None:
    st.subheader(f"Food log — {log_date.strftime('%d %b %Y')}")
    entries = log.entries(log_date)
    if not entries:
        st.info("Nothing logged for this date yet. Add a food above. A day with no entries is treated as unknown, not as zero intake.")
        return
    intakes = {i.entry_id: i for i in compute_intakes(entries, db)}
    for meal in MEALS:
        meal_entries = [e for e in entries if e.meal == meal]
        if not meal_entries:
            continue
        summary = summarize([intakes[e.entry_id] for e in meal_entries])
        st.markdown(f"**{meal}** — {summary.known_total_ug:.2f} µg" + (" (lower bound; some foods unknown)" if summary.is_lower_bound else ""))
        for entry in meal_entries:
            row = intakes[entry.entry_id]
            st.markdown(f"{row.food_name} · {row.portion_g:.0f} g · **{format_ug(row.b12_ug)}** "
                        f'<span class="source-badge">{entry.source}</span>'
                        + (f' <span class="source-badge">{row.b12_basis.replace("_", " ")}</span>' if row.b12_basis != "database" else ""),
                        unsafe_allow_html=True)
            with st.expander("Edit / remove", expanded=False):
                _render_entry_editor(db, entry, log)
    with st.expander("Reset this day's log"):
        confirmed = st.checkbox("I understand this deletes every entry for this date.", key=f"fi_clear_ok_{log_date}")
        if st.button("Clear this day", disabled=not confirmed, key=f"fi_clear_{log_date}"):
            log.clear_day(log_date)
            st.rerun()


def _render_dashboard(db: NutritionDatabase, log: FoodLog, log_date: date, preference: str, reference_key: str) -> None:
    reference = get_reference(reference_key)
    intakes = compute_intakes(log.entries(), db)
    window = filter_window(intakes, log_date, WINDOW_DAYS)
    st.subheader("Daily and weekly dashboard")
    if not window:
        st.info("The dashboard appears once you log foods. No demonstration data is shown.")
        return
    by_day = summarize_by_day(window)
    today = by_day.get(log_date)
    analysis = analyze_period(intakes, reference, log_date, WINDOW_DAYS)
    meals_today = {i.meal for i in window if i.log_date == log_date}

    m = st.columns(5)
    m[0].metric("Estimated B12 today", "no entries" if today is None else f"{today.known_total_ug:.2f} µg" + ("*" if today.is_lower_bound else ""))
    m[1].metric("Foods logged today", 0 if today is None else today.n_entries)
    m[2].metric("Meals recorded today", len(meals_today))
    m[3].metric("Data completeness", analysis.completeness.level.capitalize())
    avg = analysis.window_average
    m[4].metric(f"Average ({avg.recorded_days} recorded day{'s' if avg.recorded_days != 1 else ''})", format_ug(avg.average_ug))
    if today is not None and today.is_lower_bound:
        st.caption("* Lower bound: some logged foods have unknown B12 content (" + ", ".join(today.unknown_foods) + ").")

    c1, c2 = st.columns(2)
    with c1:
        fig = charts.food_contribution_chart(today) if today else None
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG) if fig else st.info("No B12 contributions to chart for this date.")
    with c2:
        fig = charts.meal_comparison_chart(summarize_by_meal([i for i in window if i.log_date == log_date]))
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG) if fig else st.info("No meals with known B12 on this date.")
    c3, c4 = st.columns(2)
    with c3:
        fig = charts.daily_trend_chart(by_day, log_date, WINDOW_DAYS, reference.value_ug_per_day)
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG) if fig else st.info("No recorded days in the last 7 days.")
    with c4:
        fig = charts.natural_vs_fortified_chart(summarize([i for i in window if i.log_date == log_date])) if today else None
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG) if fig else st.info("Natural vs fortified split needs known, non-zero B12 intake on this date.")

    _render_adequacy(analysis, window, reference, log_date, preference)


def _render_adequacy(analysis, window, reference, log_date: date, preference: str) -> None:
    st.subheader("Dietary intake adequacy")
    st.caption(f"Compared with: {reference.label}. {reference.note}")
    for title, comparison in (("Selected date", analysis.today), (f"Average of recorded days (last {WINDOW_DAYS} days)", analysis.window)):
        st.markdown(f"**{title}** — {comparison.label}")
        if comparison.estimated_ug is None:
            st.caption("Not enough data for this comparison.")
            continue
        a, b, c, d = st.columns(4)
        a.metric("Estimated intake", f"{comparison.estimated_ug:.2f} µg")
        b.metric("Reference", f"{reference.value_ug_per_day:g} µg")
        c.metric("Difference", f"{comparison.difference_ug:+.2f} µg")
        d.metric("% of reference", f"{comparison.percent_of_reference:.0f}%")
    st.caption(f"Days recorded in window: {analysis.completeness.recorded_days}. " + " ".join(analysis.window.caveats))

    findings = find_patterns(window, reference, log_date, preference, WINDOW_DAYS)
    indicator = screening_indicator(analysis, findings)

    st.subheader("Dietary pattern findings")
    if findings:
        for finding in findings:
            with st.expander(f"{'⚠️' if finding.severity == 'attention' else 'ℹ️'} {finding.title}", expanded=finding.severity == "attention"):
                st.write(finding.explanation)
                st.caption("Why this was flagged: " + " ".join(finding.evidence))
    else:
        st.write("No dietary pattern points were flagged from the recorded entries.")

    st.markdown(f'<div class="indicator-chip indicator-{indicator.level}">{INDICATOR_NAME}: {indicator.label}</div>', unsafe_allow_html=True)
    st.caption(indicator.explanation)

    st.subheader("Educational recommendations")
    st.markdown("\n".join(f"- {tip}" for tip in build_recommendations(analysis, findings, preference)))
    with st.expander("What food logs cannot show"):
        st.markdown("\n".join(f"- {text}" for text in RISK_FACTOR_EDUCATION))
    st.caption(DISCLAIMER)


def render_food_page() -> None:
    """Entry point used by app.py."""
    render_hero()
    try:
        db = load_database()
    except Exception as error:  # malformed or missing CSV: fail loudly rather than guess nutrient values
        st.error(f"The nutrition database could not be loaded: {error}")
        return
    log = get_session_log(st.session_state)
    log_date, preference, reference_key = render_settings()
    nonce = st.session_state.get("fi_nonce", 0)

    if flash := st.session_state.pop("fi_flash", None):
        st.success(flash)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Add food</div>', unsafe_allow_html=True)
    meal = st.selectbox("Meal", MEALS, key="fi_meal")
    photo_tab, manual_tab = st.tabs(["📷 From a photo", "✍️ Manual selection"])
    with photo_tab:
        _render_photo_tab(db, meal, log_date, log, nonce)
    with manual_tab:
        _render_manual_tab(db, meal, log_date, log, nonce)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    _render_day_log(db, log, log_date)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    _render_dashboard(db, log, log_date, preference, reference_key)
    st.markdown("</div>", unsafe_allow_html=True)
