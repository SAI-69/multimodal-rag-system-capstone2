"""
dev_view.py
───────────
Dev role — query + answer + debug panel (chunks, SQL, warnings, raw JSON).
Each debug section renders only when data for it exists.
"""

import requests
import streamlit as st
from collections import Counter
import html

from components import (
    API_BASE,
    CHUNK_COLOURS,
    answer_bubble,
    error_card,
    neon_header,
    question_bubble,
    render_image_chunk,
    render_inline_images,
    route_badge,
    small_label,
)

_HISTORY_KEY = "history_dev"


# ─────────────────────────────────────────────────────────────────────────────
# API CALL
# ─────────────────────────────────────────────────────────────────────────────
def _send_query(query: str) -> None:
    try:
        resp = requests.post(
            f"{API_BASE}/query",
            json={"query": query, "k": 5},
            timeout=120,
        )
        if resp.status_code == 200:
            st.session_state[_HISTORY_KEY].append(
                {"q": query, "data": resp.json()}
            )
        else:
            st.session_state[_HISTORY_KEY].append(
                {"q": query, "data": {"error": resp.text}}
            )
    except requests.exceptions.RequestException as exc:
        st.session_state[_HISTORY_KEY].append(
            {"q": query, "data": {"error": str(exc)}}
        )


# ─────────────────────────────────────────────────────────────────────────────
# CHUNK RENDERERS
# ─────────────────────────────────────────────────────────────────────────────
def _render_text_chunk(chunk: dict) -> None:
    st.markdown(
        f"""
        <pre style="
            background:rgba(0,0,0,0.4);
            border:1px solid rgba(255,255,255,0.05);
            border-radius:12px;
            padding:16px;
            font-size:0.8rem;
            color:rgba(255,255,255,0.85);
            white-space:pre-wrap;
            word-break:break-word;
            margin:8px 0 20px 0;
            line-height:1.7;
        ">{chunk.get("content_preview","")}</pre>
        """,
        unsafe_allow_html=True,
    )


def _render_table_chunk(chunk: dict) -> None:
    small_label("Table preview")

    raw = chunk.get("content_preview", "")

    # 🔓 Fully unescape (handles multiple layers safely)
    prev = None
    while raw != prev:
        prev = raw
        raw = html.unescape(raw)

    # ✅ If it's HTML, render in iframe (immune to Streamlit escaping)
    if "<table" in raw.lower():
        components.html(
            f"""
            <html>
              <head>
                <style>
                  body {{
                    margin: 0;
                    background: white;
                    font-family: Outfit, Arial, sans-serif;
                  }}
                  table {{
                    border-collapse: collapse;
                    width: 100%;
                    font-size: 13px;
                    color: black;
                  }}
                  th {{
                    background: #f1f5f9;
                    font-weight: 600;
                  }}
                  th, td {{
                    border: 1px solid #d1d5db;
                    padding: 8px 12px;
                    text-align: left;
                    color: black;
                  }}
                </style>
              </head>
              <body>
                {raw}
              </body>
            </html>
            """,
            height=420,
            scrolling=True,
        )
        return

    # ── Fallback: show escaped text safely
    st.code(raw)


# ─────────────────────────────────────────────────────────────────────────────
# DEBUG PANEL SECTIONS
# ─────────────────────────────────────────────────────────────────────────────
def _render_chunks(sources: list) -> None:
    with st.expander(f"Structural Databanks ({len(sources)} Nodes)", expanded=False):

        # ── Chunk type summary ───────────────────────────────────────────────
        type_counts = Counter(c.get("chunk_type", "text") for c in sources)
        small_label("Retrieved context summary")
        for t, n in type_counts.items():
            st.markdown(f"- **{t.upper()}**: {n}")

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # ── Individual chunks ───────────────────────────────────────────────
        for chunk in sources:
            ctype = chunk.get("chunk_type", "text")
            colour = CHUNK_COLOURS.get(ctype, "rgba(255,255,255,0.4)")

            st.markdown(
                f"""
                <div style="display:flex;gap:12px;align-items:center;margin:10px 0;">
                    <span style="color:rgba(255,255,255,0.3);
                                 font-size:0.7rem;
                                 font-family:JetBrains Mono,monospace;">
                        [{chunk.get("rank","")}]
                    </span>
                    <span style="
                        color:{colour};
                        font-size:0.65rem;
                        border:1px solid {colour};
                        border-radius:20px;
                        padding:2px 10px;
                        text-transform:uppercase;
                        letter-spacing:0.1em;
                        box-shadow:0 0 10px {colour}44;">
                        {ctype}
                    </span>
                    <span style="color:rgba(255,255,255,0.6);font-size:0.8rem;">
                        Sec {chunk.get("section") or "N/A"} : Pg {chunk.get("page_number","")}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if ctype == "image":
                render_image_chunk(chunk)
            elif ctype == "table":
                _render_table_chunk(chunk)
            else:
                _render_text_chunk(chunk)


def _render_sql(sql_meta) -> None:
    with st.expander("Terminal Access // SQL", expanded=False):
        if sql_meta is None:
            st.info("No SQL route was used for this query.")
            return

        small_label("Natural language inference")
        st.markdown(
            f"""
            <p style="color:rgba(255,255,255,0.8);
                      font-size:0.9rem;
                      margin:6px 0 20px;
                      line-height:1.6;
                      font-style:italic;">
            "{sql_meta.get("sql_query","")}"
            </p>
            """,
            unsafe_allow_html=True,
        )

        small_label("Execution string")
        st.code(sql_meta.get("executed_sql",""), language="sql")

        col1, col2 = st.columns(2)

        with col1:
            small_label("Records returned")
            st.markdown(
                f"""
                <p style="color:var(--accent-secondary);
                          font-size:2rem;
                          margin:0;">
                {sql_meta.get("row_count",0)}
                </p>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            small_label("Data output preview")
            st.code(sql_meta.get("result_preview",""))


def _render_warnings(warnings: list) -> None:
    with st.expander("System Diagnostics (Warnings)", expanded=True):
        for w in warnings:
            st.markdown(
                f"""
                <div style="
                    background:rgba(197,161,101,0.1);
                    border-left:3px solid #c5a165;
                    border-radius:12px;
                    padding:14px 20px;
                    font-size:0.85rem;
                    color:#e0cdab;
                    margin:8px 0;">
                {w}
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_raw_response(data: dict) -> None:
    with st.expander("Raw API Response (JSON)", expanded=False):
        st.json(data)


def _render_debug_panel(data: dict) -> None:
    sources = data.get("sources")
    sql_meta = data.get("sql_metadata")
    warnings = data.get("warnings")

    if not (sources or sql_meta is not None or warnings):
        return

    st.markdown(
        """
        <p style="color:rgba(255,255,255,0.3);
                  font-size:0.65rem;
                  letter-spacing:0.2em;
                  text-transform:uppercase;
                  margin:30px 0 12px;
                  font-weight:600;">
        System Diagnostics
        </p>
        """,
        unsafe_allow_html=True,
    )

    if sources:
        _render_chunks(sources)

    _render_sql(sql_meta)

    if warnings:
        _render_warnings(warnings)

    _render_raw_response(data)


# ─────────────────────────────────────────────────────────────────────────────
# ITEM RENDERER
# ─────────────────────────────────────────────────────────────────────────────
def _render_item(item: dict) -> None:
    question_bubble(item["q"])

    data = item["data"]

    if "error" in data:
        error_card(data["error"])
        return

    route = data.get("route")
    sources = data.get("sources")

    if route:
        route_badge(route)

    answer_bubble(data.get("answer",""))

    if sources:
        render_inline_images(sources)

    _render_debug_panel(data)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def render() -> None:
    neon_header("Development Protocol", "rgba(255,255,255,0.9)")

    if _HISTORY_KEY not in st.session_state:
        st.session_state[_HISTORY_KEY] = []

    st.markdown(
        "<div style='animation: slideUpFade 0.4s forwards;'>",
        unsafe_allow_html=True
    )

    with st.form(key="dev_query_form", clear_on_submit=True):
        col_q, col_s = st.columns([7, 2])

        with col_q:
            query_text = st.text_input(
                "Query Input",
                placeholder="Initialize inquiry...",
                label_visibility="collapsed",
            )

        with col_s:
            submit_btn = st.form_submit_button("Execute")

    st.markdown("</div>", unsafe_allow_html=True)

    if submit_btn and query_text.strip():
        with st.spinner("Processing telemetry..."):
            _send_query(query_text.strip())

    st.markdown("<hr/>", unsafe_allow_html=True)

    for item in reversed(st.session_state[_HISTORY_KEY]):
        _render_item(item)