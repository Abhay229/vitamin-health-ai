"""Evaluation helpers for the research comparisons (run on your own collected data).

Compare estimated vs reference B12 intake, and measure recognition accuracy / correction rate.
No results are bundled: figures must come from a documented evaluation set.
"""

from __future__ import annotations

import math
from typing import Sequence


def intake_error(estimated: Sequence[float], reference: Sequence[float]) -> dict:
    if len(estimated) != len(reference) or not estimated:
        raise ValueError("estimated and reference must be equal-length and non-empty.")
    diffs = [e - r for e, r in zip(estimated, reference)]
    n = len(diffs)
    rel = [abs(d) / r for d, r in zip(diffs, reference) if r > 0]
    return {"n": n, "mae_ug": sum(abs(d) for d in diffs) / n, "rmse_ug": math.sqrt(sum(d * d for d in diffs) / n),
            "mean_bias_ug": sum(diffs) / n, "mape": sum(rel) / len(rel) if rel else None}


def recognition_metrics(predicted: Sequence[str], confirmed: Sequence[str]) -> dict:
    """``predicted`` = model's top label, ``confirmed`` = label after user correction."""
    if len(predicted) != len(confirmed) or not predicted:
        raise ValueError("predicted and confirmed must be equal-length and non-empty.")
    correct = sum(p == c for p, c in zip(predicted, confirmed))
    n = len(predicted)
    return {"n": n, "top1_accuracy": correct / n, "correction_rate": 1 - correct / n}
