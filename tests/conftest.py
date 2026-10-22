from datetime import date

import pytest

from src.food_intelligence.nutrition_database import NutritionDatabase


@pytest.fixture(scope="session")
def db() -> NutritionDatabase:
    return NutritionDatabase.from_csv()


@pytest.fixture
def today() -> date:
    return date(2026, 3, 10)
