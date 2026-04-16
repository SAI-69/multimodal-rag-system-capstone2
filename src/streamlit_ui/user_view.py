
"""
user_view.py
────────────
User role — query input, answer display, inline images.
"""

import requests
import streamlit as st

from components import (
    API_BASE,
    answer_bubble,
    error_card,
    neon_header,
    question_bubble,
    render_inline_images,
    route_badge,
)

_HISTORY_KEY = "history_user"


def _send_query(query: str) -> None:
    try:
        resp = requests.post(
            f"{API_BASE}/query",
            json={"query": query, "k": 5},
            timeout=120,
        )
        if resp.status_code == 200:
            st.session_state[_HISTORY_KEY].append({"q": query, "data": resp.json()})
        else:
            st.session_state[_HISTORY_KEY].append(
                {"q": query, "data": {"error": resp.text}}
            )
    except requests.exceptions.RequestException as exc:
        st.session_state[_HISTORY_KEY].append(
            {"q": query, "data": {"error": str(exc)}}
        )


def _render_item(item: dict) -> None:
    question_bubble(item["q"])
    data = item["data"]

    if "error" in data:
        error_card(data["error"])
        return

    route   = data.get("route", "")
    sources = data.get("sources")

    if route:
        route_badge(route)

    answer_bubble(data.get("answer", ""))

    if sources:
        render_inline_images(sources)


def render() -> None:
    neon_header("Query Matrix", "rgba(0,0,0,0.9)")

    # ── Floating Query Bar ──────────────────────────────────────────────────
    st.markdown("<div style='animation: slideUpFade 0.4s forwards; position: relative;'>", unsafe_allow_html=True)
    with st.form(key="user_query_form", clear_on_submit=True):
        col_q, col_s = st.columns([7, 3])
        with col_q:
            query_text = st.text_input(
                "Query Input",
                placeholder="Initialize inquiry...",
                label_visibility="collapsed"
            )
        with col_s:
            submitted = st.form_submit_button("Transmit")
    st.markdown("</div>", unsafe_allow_html=True)

    if submitted and query_text.strip():
        with st.spinner("Accessing databanks..."):
            _send_query(query_text.strip())

    st.markdown("<hr/>", unsafe_allow_html=True)

    for item in reversed(st.session_state[_HISTORY_KEY]):
        _render_item(item)

