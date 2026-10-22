"""Educational, non-prescriptive dietary guidance built from the user's actual log and analysis."""

from __future__ import annotations

from src.food_intelligence.dietary_analysis import IntakeAnalysis, PatternFinding

DISCLAIMER = (
    "Educational information only. This tool estimates B12 from foods you log; it does not measure B12 in your body, "
    "does not diagnose deficiency, and is not a substitute for medical advice. It never recommends supplement doses."
)

RISK_FACTOR_EDUCATION = (
    "B12 status also depends on factors that food logs cannot see, for example age (absorption can fall in older adults), "
    "some stomach or intestinal conditions, past gastric surgery, and long-term use of certain medicines (such as metformin or "
    "acid-reducing drugs). A clinician can advise whether any of these apply to you.",
    "Actual B12 status is assessed with a blood test (serum/plasma B12, and sometimes additional markers) ordered by a qualified professional.",
)


def build_recommendations(analysis: IntakeAnalysis, findings: list[PatternFinding], preference: str = "non_vegetarian") -> list[str]:
    """Return educational suggestions driven by the analysis, findings and dietary preference."""
    if analysis.completeness.level == "none":
        return ["Log your meals (photo-assisted or manual) to receive suggestions based on what you actually eat."]
    tips: list[str] = []
    ids = {f.rule_id for f in findings}
    level = analysis.window.level
    low = level in {"below", "near"} or "no_b12_foods" in ids or "repeated_low" in ids

    if low:
        if preference == "non_vegetarian":
            tips.append("Consider adding suitable B12-containing foods to your meals, such as eggs, fish, meat or dairy, depending on your food choices.")
        elif preference == "vegetarian":
            tips.append("If you consume dairy and eggs, milk, curd and eggs can contribute dietary B12; consider whether they fit regularly into your meals.")
            tips.append("Check product labels for B12-fortified foods, which can add to intake.")
        else:
            tips.append("Plant foods do not reliably provide B12. Consider identifying reliably fortified foods (check the label) or discuss appropriate supplementation with a qualified healthcare professional.")
    elif level == "at_or_above":
        tips.append("Your recorded intake reaches the selected reference. Keep logging across several days; a high intake does not guarantee absorption or normal blood B12.")
    if "limited_fortified" in ids and preference != "non_vegetarian":
        tips.append("Fortified foods vary by brand; read the label for the B12 amount before relying on them.")
    if "missing_data" in ids:
        tips.append("For foods with unknown B12 content, enter a verified product-label value (µg per 100 g) if you have one so they can be included.")
    if analysis.completeness.level == "limited":
        tips.append("Log every meal and confirm portion sizes for several days to make the comparison more meaningful.")
    if low:
        tips.append("If your intake stays low, or if you have symptoms such as persistent tiredness, tingling or memory changes, consider a professional assessment rather than relying on diet alone.")
    return tips
