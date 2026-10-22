import pytest

from src.predict import vitamin_d_status


@pytest.mark.parametrize(
    ("value", "expected"), [(19.9, "Deficient"), (20.0, "Insufficient"), (30.1, "Sufficient")]
)
def test_vitamin_d_status_boundaries(value, expected):
    assert vitamin_d_status(value) == expected
