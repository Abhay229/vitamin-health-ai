"""Plotly figure builders for the nutrition dashboard (no Streamlit dependency).

Every builder returns ``None`` when there is no real data so the UI can show an empty state.
Values come only from the user's log; nothing is simulated.
"""

from __future__ import annotations

from datetime import date, timedelta

import plotly.graph_objects as go

from src.food_intelligence.b12_calculator import IntakeSummary
from src.food_intelligence.food_log import MEALS

PRIMARY = "#2563eb"
FORTIFIED = "#f59e0b"
GREY = "#94a3b8"
_LAYOUT = dict(margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300)


def _base(fig: go.Figure, title: str, ytitle: str | None = None) -> go.Figure:
    fig.update_layout(title=title, **_LAYOUT)
    if ytitle:
        fig.update_yaxes(title=ytitle)
    return fig


def food_contribution_chart(summary: IntakeSummary) -> go.Figure | None:
    if not summary.contributions:
        return None
    contributions = list(summary.contributions)[::-1]
    fig = go.Figure(go.Bar(x=[c.b12_ug for c in contributions], y=[c.food_name for c in contributions], orientation="h",
                           marker_color=PRIMARY, hovertemplate="%{y}: %{x:.2f} µg<extra></extra>"))
    fig.update_xaxes(title="Estimated B12 (µg)")
    return _base(fig, "B12 contribution by food")


def meal_comparison_chart(by_meal: dict[str, IntakeSummary]) -> go.Figure | None:
    logged = {m: s for m, s in by_meal.items() if not s.is_empty}
    if not logged:
        return None
    meals = [m for m in MEALS if m in logged]
    fig = go.Figure(go.Bar(x=meals, y=[logged[m].known_total_ug for m in meals], marker_color=PRIMARY,
                           hovertemplate="%{x}: %{y:.2f} µg<extra></extra>"))
    unknown = [m for m in meals if logged[m].is_lower_bound]
    if unknown:
        fig.add_annotation(text="Meals with unknown-B12 foods show lower bounds: " + ", ".join(unknown), xref="paper", yref="paper",
                           x=0, y=-0.18, showarrow=False, font=dict(size=11, color="#6b7280"))
    return _base(fig, "Meal-wise estimated B12", "µg")


def daily_trend_chart(by_day: dict[date, IntakeSummary], end_date: date, window_days: int = 7,
                      reference_ug: float | None = None) -> go.Figure | None:
    """Line over the window. Only *recorded* days carry a point; unlogged days are shown as 'no entries', never as 0."""
    days = [end_date - timedelta(days=n) for n in range(window_days - 1, -1, -1)]
    recorded = [d for d in days if d in by_day and not by_day[d].is_empty]
    if not recorded:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recorded, y=[by_day[d].known_total_ug for d in recorded], mode="lines+markers", name="Recorded days",
                             line=dict(color=PRIMARY), connectgaps=False, hovertemplate="%{x|%d %b}: %{y:.2f} µg<extra></extra>"))
    unrecorded = [d for d in days if d not in recorded]
    if unrecorded:
        fig.add_trace(go.Scatter(x=unrecorded, y=[0] * len(unrecorded), mode="markers", name="No entries (unknown, not zero)",
                                 marker=dict(symbol="x", color=GREY, size=9), hovertemplate="%{x|%d %b}: no entries<extra></extra>"))
    if reference_ug is not None:
        fig.add_hline(y=reference_ug, line_dash="dash", line_color="#16a34a", annotation_text="Selected reference")
    fig.update_xaxes(type="date", tickformat="%d %b")
    return _base(fig, "Daily estimated B12 (recorded days)", "µg")


def natural_vs_fortified_chart(summary: IntakeSummary) -> go.Figure | None:
    if summary.known_total_ug <= 0:
        return None
    fig = go.Figure(go.Pie(labels=["Naturally occurring (unfortified foods)", "Added through fortification"],
                           values=[summary.natural_ug, summary.fortified_ug], hole=0.55,
                           marker=dict(colors=[PRIMARY, FORTIFIED]), sort=False,
                           hovertemplate="%{label}: %{value:.2f} µg<extra></extra>"))
    return _base(fig, "Natural vs fortified B12")
