"""Streamlit page: Nutrition History (recorded days, trend, export/import)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from src.food_intelligence import dashboard as charts, meal_store
from src.food_intelligence.b12_calculator import compute_intakes, format_ug, summarize_by_day, window_average
from src.food_intelligence.dietary_analysis import DEFAULT_REFERENCE_KEY, get_reference
from src.food_intelligence.food_log import FoodLogError, get_session_log
from src.ui.food_page import PLOT_CONFIG, load_database


def render_history_page() -> None:
    st.markdown('<div class="hero-card"><div class="hero-eyebrow">Nutrition History</div><h1>Recorded days</h1>'
                "<p>Review what you have logged. Days without entries are unknown, not zero.</p></div>", unsafe_allow_html=True)
    db = load_database()
    log = get_session_log(st.session_state)
    st.caption("History lives in this browser session only. Use Export to keep it and Import to restore it; this is not secure multi-user storage.")

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    if len(log) == 0:
        st.info("No entries yet. Log foods on the Food Intelligence page to build your history.")
    else:
        intakes = compute_intakes(log.entries(), db)
        by_day = summarize_by_day(intakes)
        window_days = st.select_slider("Window (days)", options=[7, 14, 30], value=7)
        end = st.date_input("Window ends on", value=max(by_day), max_value=date.today(), key="hist_end")
        reference = get_reference(st.session_state.get("fi_ref", DEFAULT_REFERENCE_KEY))
        avg = window_average(by_day, end, window_days)
        a, b, c = st.columns(3)
        a.metric("Recorded days in window", f"{avg.recorded_days} / {avg.window_days}")
        b.metric("Average on recorded days", format_ug(avg.average_ug))
        c.metric("Recorded days with unknown-B12 foods", avg.days_with_unknown)
        fig = charts.daily_trend_chart(by_day, end, window_days, reference.value_ug_per_day)
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG) if fig else st.info("No recorded days in this window.")
        st.dataframe(pd.DataFrame([{
            "Date": d.isoformat(), "Foods logged": s.n_entries, "Known B12 (µg)": round(s.known_total_ug, 2),
            "Complete?": "yes" if s.is_complete else "no — unknown foods: " + ", ".join(s.unknown_foods),
        } for d, s in sorted(by_day.items(), reverse=True)]), hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Export / import")
    if len(log):
        st.download_button("Download log as CSV", pd.DataFrame(log.to_records()).to_csv(index=False).encode("utf-8"),
                           "food_log.csv", "text/csv", use_container_width=True)
    c1, c2 = st.columns(2)
    if c1.button("Save to local database", use_container_width=True, disabled=len(log) == 0):
        st.success(f"Saved {meal_store.save_log(log)} entries to database/meal_history.db.")
    if c2.button("Load from local database", use_container_width=True):
        try:
            added, skipped = log.merge_records(meal_store.load_records())
            st.success(f"Loaded {added} entries; skipped {skipped} already present.")
        except FoodLogError as error:
            st.error(f"Load failed: {error}")
    upload = st.file_uploader("Import a previously exported food_log.csv", type=["csv"], key="hist_import")
    if upload is not None and st.button("Import entries", use_container_width=True):
        try:
            frame = pd.read_csv(upload, dtype=str, keep_default_na=False)
            records = frame.to_dict("records")
            unknown = sorted({r.get("food_id", "") for r in records if r.get("food_id", "") not in db})
            if unknown:
                raise FoodLogError("Unknown food ids: " + ", ".join(unknown))
            added, skipped = log.merge_records(records)
            st.success(f"Imported {added} entries; skipped {skipped} that were already present.")
        except (FoodLogError, pd.errors.ParserError, UnicodeDecodeError) as error:
            st.error(f"Import failed: {error}")
    st.markdown("</div>", unsafe_allow_html=True)
