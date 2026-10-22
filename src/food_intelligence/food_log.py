"""Food log domain model, independent of Streamlit.

Entries store only *what* was eaten (food id + grams); B12 is always recomputed from the
nutrition database, so edits can never double count. Adding is idempotent on ``entry_id``,
which protects against Streamlit reruns re-submitting the same draft.

The ``FoodLog`` is storage-agnostic: ``to_records`` / ``from_records`` give a plain
serialisable form so a database-backed repository can replace session state later.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, replace
from datetime import date
from typing import Iterable, MutableMapping

from src.food_intelligence.b12_calculator import MAX_PORTION_G

MEALS = ("Breakfast", "Lunch", "Dinner", "Snacks")
SOURCES = ("manual", "image")
SESSION_KEY = "food_log"


class FoodLogError(ValueError):
    """Invalid log operation or entry."""


@dataclass(frozen=True)
class LogEntry:
    entry_id: str
    log_date: date
    meal: str
    food_id: str
    portion_g: float
    source: str = "manual"
    user_b12_ug_per_100g: float | None = None

    def __post_init__(self):
        if not self.entry_id:
            raise FoodLogError("entry_id is required.")
        if not isinstance(self.log_date, date):
            raise FoodLogError("log_date must be a date.")
        if self.meal not in MEALS:
            raise FoodLogError(f"meal must be one of {MEALS}.")
        if not self.food_id:
            raise FoodLogError("food_id is required.")
        if self.source not in SOURCES:
            raise FoodLogError(f"source must be one of {SOURCES}.")
        try:
            portion = float(self.portion_g)
        except (TypeError, ValueError) as error:
            raise FoodLogError("portion_g must be numeric.") from error
        if not math.isfinite(portion) or portion <= 0 or portion > MAX_PORTION_G:
            raise FoodLogError(f"portion_g must be > 0 and <= {MAX_PORTION_G:g}.")
        object.__setattr__(self, "portion_g", portion)
        if self.user_b12_ug_per_100g is not None:
            value = float(self.user_b12_ug_per_100g)
            if not math.isfinite(value) or value < 0:
                raise FoodLogError("user_b12_ug_per_100g must be finite and not negative.")
            object.__setattr__(self, "user_b12_ug_per_100g", value)


def new_entry_id() -> str:
    return uuid.uuid4().hex


def make_entry(log_date: date, meal: str, food_id: str, portion_g: float, *, source: str = "manual",
               user_b12_ug_per_100g: float | None = None, entry_id: str | None = None) -> LogEntry:
    return LogEntry(entry_id or new_entry_id(), log_date, meal, food_id, portion_g, source, user_b12_ug_per_100g)


class FoodLog:
    """Ordered collection of entries keyed by ``entry_id``."""

    def __init__(self, entries: Iterable[LogEntry] = ()):
        self._entries: dict[str, LogEntry] = {}
        for entry in entries:
            self.add(entry)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, entry_id: str) -> bool:
        return entry_id in self._entries

    def get(self, entry_id: str) -> LogEntry:
        try:
            return self._entries[entry_id]
        except KeyError as error:
            raise FoodLogError(f"No entry with id {entry_id!r}.") from error

    def add(self, entry: LogEntry) -> bool:
        """Add an entry. Returns False (and changes nothing) if the id already exists."""
        if entry.entry_id in self._entries:
            return False
        self._entries[entry.entry_id] = entry
        return True

    def update(self, entry_id: str, /, **changes) -> LogEntry:
        """Replace an entry in place (same id, same position). Never appends a second copy."""
        current = self.get(entry_id)
        if "entry_id" in changes:
            raise FoodLogError("entry_id cannot be changed.")
        updated = replace(current, **changes)  # re-runs validation
        self._entries[entry_id] = updated
        return updated

    def remove(self, entry_id: str) -> bool:
        return self._entries.pop(entry_id, None) is not None

    def clear_day(self, log_date: date) -> int:
        ids = [i for i, e in self._entries.items() if e.log_date == log_date]
        for entry_id in ids:
            del self._entries[entry_id]
        return len(ids)

    def entries(self, log_date: date | None = None, meal: str | None = None) -> list[LogEntry]:
        return [e for e in self._entries.values()
                if (log_date is None or e.log_date == log_date) and (meal is None or e.meal == meal)]

    def dates(self) -> list[date]:
        return sorted({e.log_date for e in self._entries.values()})

    # --- serialisation (future persistent storage hooks) ---------------------------------
    def to_records(self) -> list[dict]:
        return [{"entry_id": e.entry_id, "log_date": e.log_date.isoformat(), "meal": e.meal, "food_id": e.food_id,
                 "portion_g": e.portion_g, "source": e.source, "user_b12_ug_per_100g": e.user_b12_ug_per_100g}
                for e in self._entries.values()]

    @classmethod
    def from_records(cls, records: Iterable[dict]) -> "FoodLog":
        log = cls()
        log.merge_records(records)
        return log

    def merge_records(self, records: Iterable[dict]) -> tuple[int, int]:
        """Add records, skipping ids already present. Returns (added, skipped). Raises on invalid rows."""
        added = skipped = 0
        for rec in records:
            raw = rec.get("user_b12_ug_per_100g")
            blank = raw is None or (isinstance(raw, float) and math.isnan(raw)) or str(raw).strip() == ""
            try:
                entry = LogEntry(
                    str(rec["entry_id"]), date.fromisoformat(str(rec["log_date"])), str(rec["meal"]), str(rec["food_id"]),
                    float(rec["portion_g"]), str(rec.get("source") or "manual"), None if blank else float(raw))
            except (KeyError, ValueError, TypeError) as error:
                raise FoodLogError(f"Invalid log record {rec!r}: {error}") from error
            if self.add(entry):
                added += 1
            else:
                skipped += 1
        return added, skipped


def get_session_log(state: MutableMapping) -> FoodLog:
    """Fetch (or create) the working log from a Streamlit-like session mapping.

    Note: session state is per browser session and is *not* secure multi-user storage.
    """
    log = state.get(SESSION_KEY)
    if not isinstance(log, FoodLog):
        log = FoodLog()
        state[SESSION_KEY] = log
    return log
