"""Convert user-confirmed portion descriptions (servings, count, mL) to grams.

A photo cannot establish portion mass, so every conversion needs a documented basis in the
food database (serving weight or density); otherwise the result is None, never a guess.
"""

from __future__ import annotations

from src.food_intelligence.b12_calculator import MAX_PORTION_G
from src.food_intelligence.nutrition_database import FoodItem


def grams_from_servings(item: FoodItem, servings: float) -> float | None:
    if not item.has_standard_serving or servings <= 0:
        return None
    grams = item.serving_weight_g * servings
    return grams if grams <= MAX_PORTION_G else None


def grams_from_ml(item: FoodItem, millilitres: float) -> float | None:
    if not item.supports_volume_input or millilitres <= 0:
        return None
    grams = item.density_g_per_ml * millilitres
    return grams if grams <= MAX_PORTION_G else None
