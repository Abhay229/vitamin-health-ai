"""Optional OpenAI chat with an offline, prediction-aware fallback."""

from __future__ import annotations

import html
import os
from collections.abc import Callable

import streamlit as st


SYSTEM_PROMPT = (
    "You are a helpful Vitamin D health assistant in a Streamlit app. "
    "Explain predictions in simple language, provide practical lifestyle suggestions, "
    "and keep responses concise. Avoid diagnosis claims and remind users this is educational only."
)


def build_chat_context(model_input: dict, prediction: float, status: str) -> str:
    """Supply only the displayed lifestyle context required for a useful reply."""
    return (
        "Prediction context:\n"
        f"- Predicted Vitamin D: {prediction:.2f} ng/mL\n"
        f"- Status: {status}\n"
        f"- Sun exposure: {model_input['Sun_Exposure_min']:.0f} min/day\n"
        f"- Physical activity: {model_input['Physical_activity_hours_week']:.1f} hrs/week\n"
        f"- Fish intake: {model_input['Fish_intake_week']:.0f}/week\n"
        f"- Dairy intake: {model_input['Dairy_intake_week']:.0f}/week\n"
        f"- Indoor work: {model_input['Indoor_work_hours_day']:.1f} hrs/day\n"
    )


def get_api_key() -> str:
    """Read the optional key exclusively from the process environment."""
    return os.getenv("OPENAI_API_KEY", "").strip()


def get_openai_reply(api_key: str, conversation: list[dict], context: str) -> str:
    """Request the legacy deployment's compatible gpt-4o-mini configuration."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + context}] + conversation,
        temperature=0.4,
    )
    return response.choices[0].message.content or "I could not generate a response right now."


def map_chat_error(error: Exception) -> str:
    error_text = str(error).lower()
    if "insufficient_quota" in error_text or "exceeded your current quota" in error_text:
        return "OpenAI quota is exhausted. Using offline guidance for now."
    if "invalid_api_key" in error_text or "incorrect api key" in error_text:
        return "The OpenAI API key appears invalid. Using offline guidance for now."
    if "rate limit" in error_text or "too many requests" in error_text:
        return "OpenAI rate limit reached. Using offline guidance for now."
    if "timeout" in error_text:
        return "The OpenAI request timed out. Using offline guidance for now."
    return "Chatbot is temporarily unavailable. Using offline guidance for now."


def get_rule_based_reply(user_prompt: str, model_input: dict, prediction: float, status: str) -> str:
    """Return useful guidance without a network call or API key."""
    prompt_text = user_prompt.lower()
    suggestions: list[str] = []
    if model_input["Sun_Exposure_min"] < 20:
        suggestions.append("Aim for 20-30 minutes of safe daytime sunlight on most days when possible.")
    if model_input["Fish_intake_week"] + model_input["Dairy_intake_week"] < 5:
        suggestions.append("Increase Vitamin D-rich foods such as fatty fish, eggs, and fortified dairy.")
    if model_input["Physical_activity_hours_week"] < 3:
        suggestions.append("Add 3-5 hours per week of regular activity, preferably with some outdoor time.")
    if model_input["Indoor_work_hours_day"] > 8:
        suggestions.append("Break long indoor hours with short daylight breaks when possible.")
    if not suggestions:
        suggestions.append("Your current lifestyle profile looks supportive; maintain a consistent routine.")
    if any(word in prompt_text for word in ["30", "above 30", "sufficient", "improve"]):
        focus = "To move above 30 ng/mL, focus on consistent sunlight and diet changes for 8-12 weeks."
    elif any(word in prompt_text for word in ["food", "diet", "eat"]):
        focus = "Diet focus: consider fatty fish 2-3 times weekly and fortified foods or alternatives."
    elif any(word in prompt_text for word in ["sun", "sunlight", "exposure"]):
        focus = "Sunlight focus: increase daytime exposure gradually while avoiding sunburn."
    else:
        focus = "Based on this profile, these are practical next steps to discuss and consider."
    return "\n".join([
        f"Current prediction: {prediction:.2f} ng/mL ({status}).", focus, "",
        "Recommended actions:", *[f"- {suggestion}" for suggestion in suggestions], "",
        "Educational guidance only. Consult a clinician for diagnosis or supplements.",
    ])


def generate_reply(
    user_prompt: str,
    conversation: list[dict],
    model_input: dict,
    prediction: float,
    status: str,
    api_key: str | None = None,
    reply_provider: Callable[[str, list[dict], str], str] = get_openai_reply,
) -> str:
    """Use OpenAI when configured; otherwise reliably return offline guidance."""
    configured_key = get_api_key() if api_key is None else api_key.strip()
    if not configured_key:
        return "OpenAI API key not found. Using offline guidance.\n\n" + get_rule_based_reply(
            user_prompt, model_input, prediction, status
        )
    try:
        return reply_provider(configured_key, conversation[-10:], build_chat_context(model_input, prediction, status))
    except Exception as error:  # Provider exceptions vary by installed SDK version.
        return map_chat_error(error) + "\n\n" + get_rule_based_reply(user_prompt, model_input, prediction, status)


def init_chat_state() -> None:
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [{
            "role": "assistant",
            "content": "Hi. I am your Vitamin D assistant. Ask about your prediction, status, or lifestyle changes.",
        }]


def render_chat_widget(model_input: dict, prediction: float, status: str) -> None:
    """Render the preserved contextual chat UI once a prediction exists."""
    init_chat_state()
    with st.expander("💬 Ask the Vitamin D assistant", expanded=False):
        st.markdown(
            f'<div class="chat-context-chip">Prediction context: {prediction:.2f} ng/mL ({html.escape(status)})</div>',
            unsafe_allow_html=True,
        )
        if st.button("Clear Chat", use_container_width=True, key="clear_chat_widget"):
            st.session_state["chat_messages"] = [{"role": "assistant", "content": "Chat cleared. Ask a new question about your Vitamin D report."}]
            st.rerun()
        rows = []
        for message in st.session_state["chat_messages"][-8:]:
            role = "You" if message["role"] == "user" else "Assistant"
            role_class = "chat-msg-user" if message["role"] == "user" else "chat-msg-assistant"
            content = html.escape(message["content"]).replace("\n", "<br>")
            rows.append(f'<div class="chat-msg {role_class}"><div class="chat-role">{role}</div>{content}</div>')
        st.markdown(f'<div class="chat-log">{"".join(rows)}</div>', unsafe_allow_html=True)
        with st.form("chat_widget_form", clear_on_submit=True):
            user_prompt = st.text_input("Ask a question", placeholder="How can I move from insufficient to sufficient?", label_visibility="collapsed")
            send_clicked = st.form_submit_button("Send")
        if send_clicked and user_prompt.strip():
            prompt = user_prompt.strip()
            st.session_state["chat_messages"].append({"role": "user", "content": prompt})
            with st.spinner("Thinking..."):
                reply = generate_reply(prompt, st.session_state["chat_messages"], model_input, prediction, status)
            st.session_state["chat_messages"].append({"role": "assistant", "content": reply})
            st.rerun()
