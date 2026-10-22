from src.ui.chatbot import build_chat_context, generate_reply, get_rule_based_reply


def sample_inputs():
    return {
        "Sun_Exposure_min": 10.0, "Physical_activity_hours_week": 2.0,
        "Fish_intake_week": 1.0, "Dairy_intake_week": 2.0, "Indoor_work_hours_day": 9.0,
    }


def test_context_includes_current_prediction_and_profile():
    context = build_chat_context(sample_inputs(), 18.25, "Deficient")
    assert "18.25 ng/mL" in context
    assert "Sun exposure: 10 min/day" in context


def test_missing_key_uses_offline_guidance_without_calling_provider():
    def unexpected_provider(*args):
        raise AssertionError("provider should not be called")

    reply = generate_reply("What should I eat?", [], sample_inputs(), 18.0, "Deficient", api_key="", reply_provider=unexpected_provider)
    assert "offline guidance" in reply.lower()
    assert "Educational guidance only" in reply


def test_provider_error_falls_back_without_network():
    def unavailable(*args):
        raise RuntimeError("rate limit")

    reply = generate_reply("help", [], sample_inputs(), 18.0, "Deficient", api_key="test-key", reply_provider=unavailable)
    assert "rate limit" in reply.lower()
    assert "Recommended actions:" in reply


def test_rule_based_reply_handles_diet_question():
    reply = get_rule_based_reply("What food should I eat?", sample_inputs(), 18.0, "Deficient")
    assert "Diet focus" in reply
