"""Smoke tests with Streamlit's AppTest: navigation, manual logging, no API key, Vitamin D page intact."""

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest

from src.predict import MODEL_PATH
from src.preprocessing import PROJECT_ROOT

APP = str(PROJECT_ROOT / "app.py")


def run(page=None):
    at = AppTest.from_file(APP, default_timeout=60)
    if page:
        at.session_state["active_page"] = page
    return at.run()


def test_food_page_renders_without_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    at = run("Food Intelligence — Vitamin B12")
    assert not at.exception
    assert any("Vitamin B12 Intake Estimation" in m.value for m in at.markdown)


def test_history_page_empty_state():
    at = run("Nutrition History")
    assert not at.exception and any("No entries yet" in i.value for i in at.info)


def test_manual_logging_flow_without_vision_model():
    at = run("Food Intelligence — Vitamin B12")
    selector = next(s for s in at.selectbox if s.key and s.key.startswith("fi_manual_"))
    selector.select("egg_boiled").run()
    add = next(b for b in at.button if b.label.startswith("Add to"))
    add.click().run()
    assert not at.exception
    log = at.session_state["food_log"]
    assert len(log) == 1 and log.entries()[0].food_id == "egg_boiled" and log.entries()[0].portion_g == 50
    at.run()  # a plain rerun must not duplicate the entry
    assert len(at.session_state["food_log"]) == 1


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Vitamin D model not trained (python -m src.train)")
def test_vitamin_d_page_still_predicts():
    at = run()
    assert not at.exception
    next(b for b in at.button if b.label == "Predict Vitamin D").click().run()
    assert not at.exception and any("ng/mL" in m.value for m in at.markdown)


def test_dashboard_renders_with_logged_data_including_unknown_foods():
    from datetime import date, timedelta
    from src.food_intelligence.food_log import FoodLog, make_entry

    today = date.today()
    log = FoodLog([
        make_entry(today, "Breakfast", "egg_boiled", 100), make_entry(today, "Lunch", "paneer", 80),
        make_entry(today, "Dinner", "rice_cooked", 150), make_entry(today, "Snacks", "fortified_cereal", 30, user_b12_ug_per_100g=3.0),
        make_entry(today - timedelta(days=2), "Lunch", "milk_whole", 244),
    ])
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["active_page"] = "Food Intelligence — Vitamin B12"
    at.session_state["food_log"] = log
    at.run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "Dietary screening indicator" in text and "not a diagnosis" in text.lower()
    assert "unknown" in text.lower()  # paneer is flagged unknown, not zero
    next(b for b in at.button if b.label == "Remove entry").click().run()
    assert not at.exception and len(at.session_state["food_log"]) == 4
