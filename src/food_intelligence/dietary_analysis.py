"""Dietary-reference comparison, completeness checks, pattern rules and a screening indicator.

Everything here describes *estimated dietary intake from the user's own food log*.
Nothing here measures or predicts blood B12 status, and none of the labels is a diagnosis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from src.food_intelligence.b12_calculator import (
    EntryIntake, IntakeSummary, WindowAverage, filter_window, summarize_by_day, window_average,
)
from src.food_intelligence.food_log import MEALS

# ---------------------------------------------------------------------------------------
# Dietary references (configurable). Values in micrograms per day.
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DietaryReference:
    key: str
    label: str
    value_ug_per_day: float
    source: str
    note: str


_NIH = "NIH Office of Dietary Supplements, Vitamin B12 fact sheet (US RDA/AI)"
REFERENCES: dict[str, DietaryReference] = {r.key: r for r in (
    DietaryReference("us_adult", "US adults (14+ years): 2.4 µg/day", 2.4, _NIH,
                     "US reference, not a universal requirement for every person."),
    DietaryReference("us_pregnancy", "US pregnancy: 2.6 µg/day", 2.6, _NIH, "US reference for pregnancy."),
    DietaryReference("us_lactation", "US lactation: 2.8 µg/day", 2.8, _NIH, "US reference for lactation."),
    DietaryReference("us_9_13", "US children 9–13 years: 1.8 µg/day", 1.8, _NIH, "US reference."),
    DietaryReference("us_4_8", "US children 4–8 years: 1.2 µg/day", 1.2, _NIH, "US reference."),
    DietaryReference("us_1_3", "US children 1–3 years: 0.9 µg/day", 0.9, _NIH, "US reference."),
    DietaryReference("icmr_adult", "India ICMR-NIN adults: 2.2 µg/day", 2.2,
                     "ICMR-NIN Nutrient Requirements for Indians (2020)",
                     "Indian adult reference (ICMR-NIN 2020). Verify the exact value against the ICMR-NIN publication before citing it in formal reports."),
)}
DEFAULT_REFERENCE_KEY = "us_adult"

# Share of the reference that separates 'below' from 'near'.
NEAR_THRESHOLD = 0.75
MIN_DAYS_FOR_PATTERNS = 3  # days of logging before multi-day statements are made
MIN_MEALS_FOR_COMPLETE_DAY = 3

LEVEL_LABELS = {
    "below": "Below the selected dietary reference",
    "near": "Near the selected dietary reference",
    "at_or_above": "At or above the selected dietary reference",
    "insufficient_data": "Not enough data for a comparison",
}

PREFERENCES = {
    "non_vegetarian": "Non-vegetarian (includes meat/fish)",
    "vegetarian": "Vegetarian (dairy and/or eggs)",
    "vegan": "Vegan (no animal products)",
}


def get_reference(key: str) -> DietaryReference:
    try:
        return REFERENCES[key]
    except KeyError as error:
        raise KeyError(f"Unknown dietary reference {key!r}.") from error


# ---------------------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DataCompleteness:
    level: str  # 'none' | 'limited' | 'adequate'
    recorded_days: int
    reasons: tuple[str, ...]

    @property
    def is_sufficient(self) -> bool:
        return self.level == "adequate"


def assess_completeness(by_day: dict[date, IntakeSummary], intakes: list[EntryIntake],
                        min_days: int = MIN_DAYS_FOR_PATTERNS) -> DataCompleteness:
    """Judge whether the available log is complete enough for a meaningful comparison."""
    days = [d for d, s in by_day.items() if not s.is_empty]
    if not days:
        return DataCompleteness("none", 0, ("No foods have been logged in the selected period.",))
    reasons: list[str] = []
    if len(days) < min_days:
        reasons.append(f"Only {len(days)} day(s) recorded; at least {min_days} are needed before an average is meaningful. "
                       "Unlogged days are not counted as zero intake.")
    unknown = sorted({i.food_name for i in intakes if i.b12_ug is None})
    if unknown:
        reasons.append("B12 content is unknown for: " + ", ".join(unknown) + ". Totals are therefore lower bounds.")
    sparse = []
    for day in days:
        meals = {i.meal for i in intakes if i.log_date == day}
        if len(meals) < MIN_MEALS_FOR_COMPLETE_DAY:
            sparse.append(day.isoformat())
    if sparse:
        reasons.append(f"Fewer than {MIN_MEALS_FOR_COMPLETE_DAY} meal types were logged on: " + ", ".join(sparse) + ".")
    level = "adequate" if not reasons else "limited"
    return DataCompleteness(level, len(days), tuple(reasons))


# ---------------------------------------------------------------------------------------
# Reference comparison
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class IntakeComparison:
    reference: DietaryReference
    estimated_ug: float | None
    difference_ug: float | None
    percent_of_reference: float | None
    level: str
    label: str
    recorded_days: int
    completeness: DataCompleteness
    caveats: tuple[str, ...]


def compare_to_reference(estimated_ug: float | None, reference: DietaryReference, completeness: DataCompleteness,
                         near_threshold: float = NEAR_THRESHOLD) -> IntakeComparison:
    if estimated_ug is not None and (not math.isfinite(estimated_ug) or estimated_ug < 0):
        raise ValueError("estimated_ug must be finite and not negative.")
    caveats = list(completeness.reasons)
    caveats.append("Dietary intake estimates do not measure B12 status in the body; absorption varies between people.")
    if estimated_ug is None or completeness.level == "none":
        return IntakeComparison(reference, None, None, None, "insufficient_data", LEVEL_LABELS["insufficient_data"],
                                completeness.recorded_days, completeness, tuple(caveats))
    ratio = estimated_ug / reference.value_ug_per_day
    level = "at_or_above" if ratio >= 1.0 else "near" if ratio >= near_threshold else "below"
    return IntakeComparison(reference, estimated_ug, estimated_ug - reference.value_ug_per_day, ratio * 100.0,
                            level, LEVEL_LABELS[level], completeness.recorded_days, completeness, tuple(caveats))


@dataclass(frozen=True)
class IntakeAnalysis:
    today: IntakeComparison
    window: IntakeComparison
    window_average: WindowAverage
    completeness: DataCompleteness


def analyze_period(intakes: list[EntryIntake], reference: DietaryReference, end_date: date,
                   window_days: int = 7) -> IntakeAnalysis:
    """Compare *today* and the *average of recorded days in the window* with the reference."""
    in_window = filter_window(intakes, end_date, window_days)
    by_day = summarize_by_day(in_window)
    avg = window_average(by_day, end_date, window_days)
    completeness = assess_completeness(by_day, in_window)
    today_summary = by_day.get(end_date)
    today_completeness = (
        DataCompleteness("none", 0, ("Nothing has been logged for this date.",)) if today_summary is None
        else DataCompleteness(
            "adequate" if today_summary.is_complete and len({i.meal for i in in_window if i.log_date == end_date}) >= MIN_MEALS_FOR_COMPLETE_DAY else "limited",
            1,
            ("One day of logging is not a measure of long-term intake.",)
            + (("B12 content is unknown for: " + ", ".join(today_summary.unknown_foods) + ".",) if today_summary.unknown_foods else ()),
        )
    )
    today = compare_to_reference(None if today_summary is None else today_summary.known_total_ug, reference, today_completeness)
    # Averages need enough days; fewer days are still reported but are clearly flagged by completeness.
    window = compare_to_reference(avg.average_ug, reference, completeness)
    return IntakeAnalysis(today, window, avg, completeness)


# ---------------------------------------------------------------------------------------
# Dietary pattern findings (transparent rules)
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PatternFinding:
    rule_id: str
    title: str
    explanation: str
    evidence: tuple[str, ...]
    severity: str  # 'info' | 'attention'


_LIMITATION = "Food logging cannot determine whether your body has adequate B12 stores."


def find_patterns(intakes: list[EntryIntake], reference: DietaryReference, end_date: date,
                  preference: str = "non_vegetarian", window_days: int = 7) -> list[PatternFinding]:
    """Apply explicit rules to the logged entries in the window. Every finding lists its evidence."""
    if preference not in PREFERENCES:
        raise ValueError(f"Unknown dietary preference {preference!r}.")
    in_window = filter_window(intakes, end_date, window_days)
    if not in_window:
        return []
    by_day = summarize_by_day(in_window)
    days = sorted(by_day)
    findings: list[PatternFinding] = []

    known_positive = [i for i in in_window if i.b12_ug is not None and i.b12_ug > 0]
    if not known_positive:
        findings.append(PatternFinding(
            "no_b12_foods", "Few identified B12-containing foods",
            "Your recorded meals contain few identified B12-containing or fortified foods. This suggests that your "
            "estimated dietary intake may be low. " + _LIMITATION,
            (f"{len(in_window)} logged item(s); none with a known B12 content above 0 µg.",), "attention"))

    animal_days = {i.log_date for i in in_window if i.is_animal_source and preference != "vegan"}
    if preference != "vegan":
        flesh = {"fish", "shellfish", "poultry", "red_meat", "organ_meat"}
        if preference == "non_vegetarian":
            relevant_days = {i.log_date for i in in_window if i.category in flesh}
            what = "fish, meat, poultry or shellfish"
        else:
            relevant_days = animal_days
            what = "dairy or egg"
        if len(relevant_days) / len(days) < 0.5:
            findings.append(PatternFinding(
                "limited_animal_source", "Limited recorded animal-source foods",
                f"Few of your recorded days include {what}, which are the main natural dietary sources of B12 for your selected "
                "preference. This is a description of the log, not a deficiency finding.",
                (f"{what} recorded on {len(relevant_days)} of {len(days)} recorded day(s).",), "attention"))

    fortified_days = {i.log_date for i in in_window if i.is_fortified}
    if not fortified_days:
        if preference == "vegan":
            findings.append(PatternFinding(
                "no_fortified_vegan", "No fortified foods recorded (vegan preference)",
                "Plant foods do not naturally provide reliable B12, so people following a vegan diet usually depend on "
                "fortified foods or supplements. None were recorded. This does not mean you are deficient; you may "
                "simply not have logged them.",
                ("No fortified products in the logged period.",), "attention"))
        else:
            findings.append(PatternFinding(
                "limited_fortified", "Limited recorded fortified foods",
                "No B12-fortified foods were recorded. Fortified foods are one optional contributor to intake.",
                ("No fortified products in the logged period.",), "info"))

    complete_low = [d for d in days if by_day[d].is_complete and by_day[d].known_total_ug < reference.value_ug_per_day]
    if len(days) >= MIN_DAYS_FOR_PATTERNS and len(complete_low) >= max(2, math.ceil(len(days) / 2)):
        findings.append(PatternFinding(
            "repeated_low", "Repeatedly low estimated intake",
            f"Estimated intake was below the selected reference ({reference.value_ug_per_day:g} µg/day) on several recorded "
            "days. " + _LIMITATION,
            (f"{len(complete_low)} of {len(days)} recorded day(s) below reference: " + ", ".join(d.isoformat() for d in complete_low),),
            "attention"))

    unknown = sorted({i.food_name for i in in_window if i.b12_ug is None})
    if unknown:
        findings.append(PatternFinding(
            "missing_data", "Missing nutrient or portion information",
            "Some logged foods have no verified B12 value in the local database, so they are counted as UNKNOWN (not as zero). "
            "Enter a product-label value where one exists.",
            ("Unknown B12: " + ", ".join(unknown),), "info"))

    sparse = [d.isoformat() for d in days if len({i.meal for i in in_window if i.log_date == d}) < MIN_MEALS_FOR_COMPLETE_DAY]
    if sparse:
        findings.append(PatternFinding(
            "incomplete_meals", "Some days have few logged meals",
            "Days with only one or two meals logged may under-represent what you actually ate.",
            ("Days with fewer than 3 meal types: " + ", ".join(sparse),), "info"))
    return findings


# ---------------------------------------------------------------------------------------
# Rule-based dietary screening indicator (NOT a clinical probability)
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ScreeningIndicator:
    level: str  # 'insufficient_data' | 'review' | 'no_flag'
    label: str
    explanation: str


INDICATOR_NAME = "Dietary screening indicator"


def screening_indicator(analysis: IntakeAnalysis, findings: list[PatternFinding]) -> ScreeningIndicator:
    """A transparent, rule-based summary of the *log*. It is not a deficiency probability or diagnosis."""
    tail = " This is a dietary screening indicator, not a diagnosis; serum B12 testing and clinical assessment determine actual status."
    if analysis.completeness.level == "none":
        return ScreeningIndicator("insufficient_data", "Not enough logged data", "Log some meals to see an indicator." + tail)
    window = analysis.window
    attention = [f for f in findings if f.severity == "attention"]
    if window.level in {"below", "near"} and (attention or window.level == "below"):
        return ScreeningIndicator(
            "review", "Lower dietary-intake pattern in the recorded log",
            f"The recorded log is {window.label.lower()} and {len(attention)} dietary point(s) are flagged for review." + tail)
    if window.level == "at_or_above" and not attention:
        return ScreeningIndicator(
            "no_flag", "No lower-intake pattern flagged in the recorded log",
            "The recorded log reaches the selected reference and no dietary points were flagged. This does not guarantee adequate absorption or normal blood B12." + tail)
    return ScreeningIndicator("review", "Some dietary points to review", "Some dietary points in the recorded log may be worth reviewing." + tail)
