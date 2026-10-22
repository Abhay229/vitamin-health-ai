"""Streamlit-independent Vitamin B12 intake calculations.

Core formula::

    B12 (ug) = portion mass (g) * B12 content (ug per 100 g) / 100

Rules enforced here
-------------------
* Missing nutrient data is ``None`` (unknown) and is never silently treated as zero.
* Rounding is a display concern only (``format_ug``); internal values keep full precision.
* Intake is always *recomputed* from log entries + the database, so editing an
  entry can never double count.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Iterable, Protocol

from src.food_intelligence.nutrition_database import NutritionDatabase

MAX_PORTION_G = 5000.0
B12_UNIT = "µg"  # micrograms


class InvalidPortionError(ValueError):
    """Portion mass/volume is negative, zero, non-finite, or implausibly large."""


class InvalidNutrientValueError(ValueError):
    """Nutrient density is negative or non-finite."""


def _validate_portion(portion_g: float) -> float:
    try:
        value = float(portion_g)
    except (TypeError, ValueError) as error:
        raise InvalidPortionError(f"Portion must be a number of grams, got {portion_g!r}.") from error
    if not math.isfinite(value) or value <= 0:
        raise InvalidPortionError("Portion weight must be a positive, finite number of grams.")
    if value > MAX_PORTION_G:
        raise InvalidPortionError(f"Portion weight above {MAX_PORTION_G:g} g is not accepted for one entry.")
    return value


def _validate_density(b12_ug_per_100g: float) -> float:
    try:
        value = float(b12_ug_per_100g)
    except (TypeError, ValueError) as error:
        raise InvalidNutrientValueError(f"B12 value must be numeric, got {b12_ug_per_100g!r}.") from error
    if not math.isfinite(value) or value < 0:
        raise InvalidNutrientValueError("B12 content (µg per 100 g) must be finite and not negative.")
    return value


def calculate_b12_ug(portion_g: float, b12_ug_per_100g: float) -> float:
    """B12 in micrograms for a portion given in grams and a density in µg/100 g."""
    return _validate_portion(portion_g) * _validate_density(b12_ug_per_100g) / 100.0


def calculate_entry_b12(portion_g: float, b12_ug_per_100g: float | None) -> float | None:
    """Like :func:`calculate_b12_ug` but returns ``None`` (unknown) when the density is missing.

    The portion is still validated so an invalid portion is never hidden by missing data.
    """
    _validate_portion(portion_g)
    if b12_ug_per_100g is None:
        return None
    return calculate_b12_ug(portion_g, b12_ug_per_100g)


def volume_ml_to_grams(volume_ml: float, density_g_per_ml: float | None) -> float:
    """Convert millilitres to grams using a *documented* density; refuses to guess."""
    if density_g_per_ml is None:
        raise InvalidPortionError("This food has no documented density; enter the portion in grams.")
    try:
        volume = float(volume_ml)
        density = float(density_g_per_ml)
    except (TypeError, ValueError) as error:
        raise InvalidPortionError("Volume and density must be numeric.") from error
    if not (math.isfinite(volume) and volume > 0 and math.isfinite(density) and density > 0):
        raise InvalidPortionError("Volume and density must be positive, finite numbers.")
    return volume * density


def format_ug(value: float | None, digits: int = 2) -> str:
    """Display helper (the only place rounding happens)."""
    return "unknown" if value is None else f"{value:.{digits}f} {B12_UNIT}"


class _EntryLike(Protocol):
    entry_id: str
    log_date: date
    meal: str
    food_id: str
    portion_g: float
    user_b12_ug_per_100g: float | None


@dataclass(frozen=True)
class EntryIntake:
    """Computed B12 for one log entry."""

    entry_id: str
    log_date: date
    meal: str
    food_id: str
    food_name: str
    category: str
    portion_g: float
    b12_ug_per_100g: float | None
    b12_ug: float | None  # None == unknown
    b12_basis: str  # 'database' | 'user_label' | 'missing'
    is_fortified: bool
    is_animal_source: bool


def entry_intake(entry: _EntryLike, db: NutritionDatabase) -> EntryIntake:
    """Compute the intake of one entry. A user-supplied label value only fills a *missing* record."""
    item = db.get(entry.food_id)
    if item.b12_ug_per_100g is not None:
        density, basis = item.b12_ug_per_100g, "database"
    elif entry.user_b12_ug_per_100g is not None:
        density, basis = _validate_density(entry.user_b12_ug_per_100g), "user_label"
    else:
        density, basis = None, "missing"
    return EntryIntake(
        entry_id=entry.entry_id, log_date=entry.log_date, meal=entry.meal, food_id=item.food_id,
        food_name=item.food_name, category=item.category, portion_g=float(entry.portion_g),
        b12_ug_per_100g=density, b12_ug=calculate_entry_b12(entry.portion_g, density), b12_basis=basis,
        is_fortified=item.is_fortified, is_animal_source=item.is_animal_source,
    )


def compute_intakes(entries: Iterable[_EntryLike], db: NutritionDatabase) -> list[EntryIntake]:
    return [entry_intake(entry, db) for entry in entries]


@dataclass(frozen=True)
class FoodContribution:
    food_id: str
    food_name: str
    b12_ug: float
    share_of_known: float | None  # fraction of the known total; None if the known total is 0


@dataclass(frozen=True)
class IntakeSummary:
    """Aggregate of a set of entries (a meal, a day, ...)."""

    known_total_ug: float
    entries_counted: int
    entries_unknown: int
    unknown_foods: tuple[str, ...]
    natural_ug: float
    fortified_ug: float
    contributions: tuple[FoodContribution, ...]
    by_meal: dict[str, float] = field(default_factory=dict)

    @property
    def n_entries(self) -> int:
        return self.entries_counted + self.entries_unknown

    @property
    def is_empty(self) -> bool:
        return self.n_entries == 0

    @property
    def is_complete(self) -> bool:
        """True when no logged food had unknown B12 content (and something was logged)."""
        return self.n_entries > 0 and self.entries_unknown == 0

    @property
    def is_lower_bound(self) -> bool:
        """When foods with unknown B12 exist, ``known_total_ug`` is only a lower bound."""
        return self.entries_unknown > 0


def summarize(intakes: Iterable[EntryIntake]) -> IntakeSummary:
    """Aggregate intakes. Unknown values are counted separately and never added as zero."""
    intakes = list(intakes)
    per_food: dict[str, list] = defaultdict(lambda: [None, 0.0])
    by_meal: dict[str, float] = defaultdict(float)
    natural = fortified = total = 0.0
    counted = unknown = 0
    unknown_foods: list[str] = []
    for item in intakes:
        if item.b12_ug is None:
            unknown += 1
            if item.food_name not in unknown_foods:
                unknown_foods.append(item.food_name)
            continue
        counted += 1
        total += item.b12_ug
        by_meal[item.meal] += item.b12_ug
        per_food[item.food_id][0] = item.food_name
        per_food[item.food_id][1] += item.b12_ug
        if item.is_fortified:
            fortified += item.b12_ug
        else:
            natural += item.b12_ug
    contributions = tuple(
        FoodContribution(fid, name, ug, (ug / total) if total > 0 else None)
        for fid, (name, ug) in sorted(per_food.items(), key=lambda kv: -kv[1][1])
    )
    return IntakeSummary(total, counted, unknown, tuple(unknown_foods), natural, fortified, contributions, dict(by_meal))


def summarize_by_meal(intakes: Iterable[EntryIntake]) -> dict[str, IntakeSummary]:
    groups: dict[str, list[EntryIntake]] = defaultdict(list)
    for item in intakes:
        groups[item.meal].append(item)
    return {meal: summarize(items) for meal, items in groups.items()}


def summarize_by_day(intakes: Iterable[EntryIntake]) -> dict[date, IntakeSummary]:
    """One summary per date that has at least one entry. Dates with no entries are absent
    (an unlogged day is *unknown*, not zero)."""
    groups: dict[date, list[EntryIntake]] = defaultdict(list)
    for item in intakes:
        groups[item.log_date].append(item)
    return {day: summarize(items) for day, items in sorted(groups.items())}


@dataclass(frozen=True)
class WindowAverage:
    average_ug: float | None  # None when no day in the window has any usable data
    recorded_days: int
    window_days: int
    days_with_unknown: int  # recorded days that contain foods with unknown B12
    start: date
    end: date


def window_average(by_day: dict[date, IntakeSummary], end_date: date, window_days: int = 7) -> WindowAverage:
    """Average known intake over *recorded* days in the window ending at ``end_date`` (inclusive).

    Days without any entry are excluded from both numerator and denominator.
    """
    if window_days < 1:
        raise ValueError("window_days must be at least 1.")
    start = end_date - timedelta(days=window_days - 1)
    recorded = [s for d, s in by_day.items() if start <= d <= end_date and not s.is_empty]
    if not recorded:
        return WindowAverage(None, 0, window_days, 0, start, end_date)
    average = sum(s.known_total_ug for s in recorded) / len(recorded)
    return WindowAverage(average, len(recorded), window_days, sum(1 for s in recorded if s.is_lower_bound), start, end_date)


def filter_window(intakes: Iterable[EntryIntake], end_date: date, window_days: int = 7) -> list[EntryIntake]:
    start = end_date - timedelta(days=window_days - 1)
    return [i for i in intakes if start <= i.log_date <= end_date]
