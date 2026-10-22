"""Local, curated food-composition database for the Vitamin B12 module.

The database is read from ``data/food_nutrition.csv``. It never calls the network.
A missing B12 value is represented as ``None`` and is never converted to zero.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.preprocessing import PROJECT_ROOT


FOOD_DATA_PATH = PROJECT_ROOT / "data" / "food_nutrition.csv"

REQUIRED_COLUMNS = (
    "food_id", "food_name", "category", "aliases", "diet_suitability", "serving_description",
    "serving_weight_g", "density_g_per_ml", "b12_ug_per_100g", "b12_value_status", "source_name",
    "source_reference", "data_version", "is_fortified", "food_preparation", "data_quality_note",
)

CATEGORIES = (
    "dairy", "egg", "fish", "shellfish", "poultry", "red_meat", "organ_meat", "fortified", "plant",
)
ANIMAL_SOURCE_CATEGORIES = frozenset({"dairy", "egg", "fish", "shellfish", "poultry", "red_meat", "organ_meat"})
DIET_SUITABILITY = ("vegan", "vegetarian", "non_vegetarian")

# Statuses that explain exactly why a value is present or absent.
VALUE_STATUSES_WITH_VALUE = ("reported", "reported_zero_plant")
VALUE_STATUSES_MISSING = ("missing_no_reliable_record", "missing_brand_dependent")
VALUE_STATUSES = VALUE_STATUSES_WITH_VALUE + VALUE_STATUSES_MISSING


class NutritionDataError(ValueError):
    """Raised when the nutrition CSV is malformed."""


@dataclass(frozen=True)
class FoodItem:
    food_id: str
    food_name: str
    category: str
    aliases: tuple[str, ...]
    diet_suitability: str
    serving_description: str | None
    serving_weight_g: float | None
    density_g_per_ml: float | None
    b12_ug_per_100g: float | None  # None means UNKNOWN, never zero
    b12_value_status: str
    source_name: str
    source_reference: str
    data_version: str
    is_fortified: bool
    food_preparation: str
    data_quality_note: str

    @property
    def has_b12_value(self) -> bool:
        return self.b12_ug_per_100g is not None

    @property
    def is_animal_source(self) -> bool:
        return self.category in ANIMAL_SOURCE_CATEGORIES

    @property
    def has_standard_serving(self) -> bool:
        return self.serving_weight_g is not None and self.serving_weight_g > 0

    @property
    def supports_volume_input(self) -> bool:
        return self.density_g_per_ml is not None and self.density_g_per_ml > 0


def _clean(value) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


def _to_float(value) -> float | None:
    text = _clean(value)
    return None if text is None else float(text)


def _to_bool(value) -> bool:
    text = (_clean(value) or "").lower()
    if text not in {"true", "false"}:
        raise ValueError(f"is_fortified must be 'true' or 'false', got {value!r}")
    return text == "true"


def validate_dataframe(frame: pd.DataFrame) -> list[str]:
    """Return a list of human-readable problems; an empty list means the data is valid."""
    problems: list[str] = []
    missing_cols = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_cols:
        return ["Missing required columns: " + ", ".join(missing_cols)]
    if frame["food_id"].isna().any() or (frame["food_id"].astype(str).str.strip() == "").any():
        problems.append("food_id must not be empty.")
    duplicated = frame.loc[frame["food_id"].duplicated(), "food_id"].tolist()
    if duplicated:
        problems.append("Duplicate food_id values: " + ", ".join(map(str, duplicated)))

    for row in frame.to_dict("records"):
        fid = row["food_id"]
        if not re.fullmatch(r"[a-z0-9_]+", str(fid)):
            problems.append(f"{fid}: food_id must be lowercase letters, digits and underscores.")
        if row["category"] not in CATEGORIES:
            problems.append(f"{fid}: unknown category {row['category']!r}.")
        if row["diet_suitability"] not in DIET_SUITABILITY:
            problems.append(f"{fid}: unknown diet_suitability {row['diet_suitability']!r}.")
        for column in ("food_name", "source_name", "source_reference", "data_version", "data_quality_note"):
            if _clean(row[column]) is None:
                problems.append(f"{fid}: {column} must not be empty (sources must be traceable).")
        try:
            fortified = _to_bool(row["is_fortified"])
        except ValueError as error:
            problems.append(f"{fid}: {error}")
            fortified = None
        if fortified is not None and (row["category"] == "fortified") != fortified:
            problems.append(f"{fid}: category 'fortified' and is_fortified must agree.")

        for column in ("serving_weight_g", "density_g_per_ml", "b12_ug_per_100g"):
            try:
                number = _to_float(row[column])
            except ValueError:
                problems.append(f"{fid}: {column} is not numeric: {row[column]!r}.")
                continue
            if number is None:
                continue
            if not math.isfinite(number):
                problems.append(f"{fid}: {column} must be finite.")
            elif column == "b12_ug_per_100g" and number < 0:
                problems.append(f"{fid}: b12_ug_per_100g must not be negative.")
            elif column != "b12_ug_per_100g" and number <= 0:
                problems.append(f"{fid}: {column} must be positive when provided.")

        status = row["b12_value_status"]
        value = _clean(row["b12_ug_per_100g"])
        if status not in VALUE_STATUSES:
            problems.append(f"{fid}: unknown b12_value_status {status!r}.")
        elif status in VALUE_STATUSES_MISSING and value is not None:
            problems.append(f"{fid}: status {status!r} requires an empty b12_ug_per_100g.")
        elif status in VALUE_STATUSES_WITH_VALUE and value is None:
            problems.append(f"{fid}: status {status!r} requires a numeric b12_ug_per_100g.")
        elif status == "reported_zero_plant" and (row["category"] != "plant" or _to_float(value) != 0.0):
            problems.append(f"{fid}: 'reported_zero_plant' is only valid for plant foods with value 0.")
        if status == "missing_brand_dependent" and fortified is False:
            problems.append(f"{fid}: brand-dependent records should be fortified products.")
    return problems


class NutritionDatabase:
    """In-memory view of the curated food table."""

    def __init__(self, frame: pd.DataFrame):
        problems = validate_dataframe(frame)
        if problems:
            raise NutritionDataError("; ".join(problems))
        self._items: dict[str, FoodItem] = {}
        for row in frame.to_dict("records"):
            aliases = tuple(a.strip().lower() for a in (_clean(row["aliases"]) or "").split(";") if a.strip())
            self._items[str(row["food_id"])] = FoodItem(
                food_id=str(row["food_id"]), food_name=str(row["food_name"]).strip(), category=row["category"],
                aliases=aliases, diet_suitability=row["diet_suitability"],
                serving_description=_clean(row["serving_description"]),
                serving_weight_g=_to_float(row["serving_weight_g"]),
                density_g_per_ml=_to_float(row["density_g_per_ml"]),
                b12_ug_per_100g=_to_float(row["b12_ug_per_100g"]),
                b12_value_status=row["b12_value_status"], source_name=str(row["source_name"]).strip(),
                source_reference=str(row["source_reference"]).strip(), data_version=str(row["data_version"]).strip(),
                is_fortified=_to_bool(row["is_fortified"]), food_preparation=_clean(row["food_preparation"]) or "",
                data_quality_note=str(row["data_quality_note"]).strip(),
            )

    @classmethod
    def from_csv(cls, path: Path | str = FOOD_DATA_PATH) -> "NutritionDatabase":
        # keep_default_na=False would turn blanks into '' (fine); we parse blanks ourselves.
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
        return cls(frame)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, food_id: str) -> bool:
        return food_id in self._items

    def get(self, food_id: str) -> FoodItem:
        try:
            return self._items[food_id]
        except KeyError as error:
            raise KeyError(f"Unknown food_id: {food_id!r}") from error

    def all_items(self) -> list[FoodItem]:
        return list(self._items.values())

    def by_category(self, category: str) -> list[FoodItem]:
        return [item for item in self._items.values() if item.category == category]

    def search(self, text: str) -> list[FoodItem]:
        """Case-insensitive match on name or alias."""
        needle = text.strip().lower()
        if not needle:
            return []
        return [i for i in self._items.values() if needle in i.food_name.lower() or any(needle in a for a in i.aliases)]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([item.__dict__ for item in self._items.values()])


def main() -> int:  # pragma: no cover - thin CLI wrapper
    """``python -m src.food_intelligence.nutrition_database`` validates the CSV."""
    frame = pd.read_csv(FOOD_DATA_PATH, dtype=str, keep_default_na=False, na_values=[""])
    problems = validate_dataframe(frame)
    if problems:
        print("Validation FAILED:")
        for problem in problems:
            print(" -", problem)
        return 1
    missing = frame["b12_ug_per_100g"].isna().sum()
    print(f"OK: {len(frame)} foods validated; {missing} records have no B12 value (reported as unknown, not zero).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
