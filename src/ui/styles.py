"""The existing deployment application's visual treatment."""

import streamlit as st


APP_CSS = """
<style>
:root { --bg:#f7f7fb; --panel:#fff; --border:#e7e8ef; --text:#111827; --muted:#6b7280; --primary:#2563eb; --primary-strong:#1d4ed8; }
.stApp { background:radial-gradient(circle at top right,#eef4ff 0%,#f7f7fb 45%,#f7f7fb 100%); color:var(--text); }
.block-container, [data-testid="stMainBlockContainer"] { max-width:1180px; padding-top:4.5rem !important; padding-bottom:1.25rem; }
[data-testid="stHeader"] { background:transparent; pointer-events:none; }
[data-testid="stHeader"] [data-testid="stToolbar"] { pointer-events:none; }
[data-testid="stToolbarActions"], [data-testid="stMainMenu"], [data-testid="stAppDeployButton"] { pointer-events:auto; }
div[role="radiogroup"][aria-label="Navigate"] label, div.stRadio > div[role="radiogroup"] > label { cursor:pointer; position:relative; z-index:5; }
div.stRadio[data-testid="stRadio"] > div[role="radiogroup"] { gap:.5rem; }
.hero-card { position:relative; overflow:hidden; background:radial-gradient(circle at top right,rgba(37,99,235,.18),transparent 34%),linear-gradient(135deg,#fff 0%,#f7fbff 52%,#eef4ff 100%); border:1px solid rgba(191,219,254,.9); border-radius:22px; padding:1rem 1.15rem; box-shadow:0 18px 42px rgba(15,23,42,.08); margin-bottom:.7rem; }
.hero-eyebrow,.mode-switch-label,.section-title { font-size:.75rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:#1d4ed8; margin-bottom:.35rem; }
.hero-card p,.mode-label { margin:0; color:#475569; line-height:1.5; }
.panel { background:var(--panel); border:1px solid var(--border); border-radius:18px; padding:1rem; box-shadow:0 10px 30px rgba(17,24,39,.06); margin-bottom:.8rem; }
.result-value { font-size:2.15rem; font-weight:800; color:var(--text); line-height:1.15; margin:.2rem 0 .35rem; }
.status-chip { display:inline-block; padding:.32rem .68rem; border-radius:999px; font-weight:600; font-size:.88rem; margin-bottom:.75rem; }
.status-deficient { background:#fef2f2; border:1px solid #fecaca; color:#b91c1c; }
.status-insufficient { background:#fffbeb; border:1px solid #fde68a; color:#a16207; }
.status-sufficient { background:#f0fdf4; border:1px solid #bbf7d0; color:#15803d; }
.interp-card { background:#f8faff; border:1px solid var(--border); border-radius:12px; padding:.75rem; min-height:104px; }
.interp-card-title { color:var(--muted); font-size:.78rem; font-weight:700; text-transform:uppercase; }
.interp-card-value { color:var(--text); font-size:1.12rem; font-weight:800; margin-top:.35rem; }
.interp-card-delta { color:#2563eb; font-weight:700; margin-top:.2rem; }
.chat-log { max-height:320px; overflow-y:auto; padding:.25rem; }
.chat-msg { border-radius:12px; padding:.6rem .7rem; margin:.45rem 0; line-height:1.4; }
.chat-msg-user { background:#eff6ff; border:1px solid #dbeafe; }
.chat-msg-assistant { background:#f8fafc; border:1px solid #e2e8f0; }
.chat-role { font-size:.72rem; font-weight:800; text-transform:uppercase; color:#64748b; margin-bottom:.2rem; }
.chat-context-chip { display:inline-block; padding:.25rem .5rem; border-radius:999px; background:#eff6ff; color:#1d4ed8; font-size:.8rem; margin-bottom:.5rem; }
.notice { background:#fffbeb; border:1px solid #fde68a; color:#92400e; border-radius:12px; padding:.55rem .75rem; font-size:.85rem; line-height:1.45; margin-bottom:.7rem; }
.source-badge { display:inline-block; padding:.1rem .45rem; border-radius:999px; background:#eff6ff; color:#1d4ed8; font-size:.72rem; font-weight:700; text-transform:uppercase; }
.indicator-chip { display:inline-block; padding:.4rem .75rem; border-radius:999px; font-weight:700; font-size:.9rem; margin:.4rem 0 .2rem; }
.indicator-review { background:#fffbeb; border:1px solid #fde68a; color:#a16207; }
.indicator-no_flag { background:#f0fdf4; border:1px solid #bbf7d0; color:#15803d; }
.indicator-insufficient_data { background:#f1f5f9; border:1px solid #cbd5e1; color:#475569; }
.stButton > button { min-height:2.75rem; border-radius:12px; font-weight:700; }
@media (max-width:768px) { .block-container { padding-left:.75rem; padding-right:.75rem; } .hero-card { border-radius:16px; } }
</style>
"""


def apply_global_styles() -> None:
    """Render the retained deployment CSS once per Streamlit rerun."""
    st.markdown(APP_CSS, unsafe_allow_html=True)
