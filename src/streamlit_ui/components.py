"""
components.py
─────────────
Shared CSS injection and primitive UI helpers for Glassmorphic Cyber Aesthetic.
"""

import base64
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000/api/v1"

CHUNK_COLOURS = {
    "text":  "#e6ff00",
    "table": "#00000000",
    "image": "#888888",
}
ROUTE_COLOURS = {
    "vector": "rgba(230, 255, 0, 0.9)",
    "rdbms":  "rgba(255, 255, 255, 0.9)",
    "hybrid": "rgba(136, 136, 136, 0.9)",
}


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
def inject_css() -> None:
    """Injects exactly Streamlit 1.56.0 compatible CSS for Glassmorphism."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

        /* ── Base Variables & Animations ─────────────────────────────────── */
        :root {
            --surface-bg: #ffffff;
            --surface-border: #e2e8f0;
            --accent-primary: #d5ec00;
            --accent-secondary: #000000;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --bg-dark: #f8fafc;
        }

        @keyframes slideUpFade {
            0% { opacity: 0; transform: translateY(12px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        html, body, [class*="css"] {
            font-family: 'Outfit', 'Helvetica Neue', sans-serif;
            font-weight: 400;
        }

        /* Ambient glowing background */
        .stApp {
            background-color: var(--bg-dark) !important;
            background-image: none !important;
            color: var(--text-main);
        }

        /* ── Hide Streamlit Chrome ───────────────────────────────────────── */
        #MainMenu, footer { visibility: hidden; }
        header { background: transparent !important; }
        header [data-testid="stToolbar"] { visibility: hidden; }
        [data-testid="stDecoration"] { display: none; }

        /* ── Scrollbar ───────────────────────────────────────────────────── */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.15); border-radius: 10px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(0,0,0,0.25); }

        /* ── Sidebar (Panel) & Toggle Controls ───────────────────────────── */
        [data-testid="stSidebar"] {
            background: #f1f5f9 !important;
            border-right: 1px solid var(--surface-border) !important;
        }
        [data-testid="stSidebar"] > div:first-child { padding-top: 30px; }

        /* All text/icons inside sidebar: force dark */
        [data-testid="stSidebar"] *,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] [data-testid="stIconMaterial"] {
            color: var(--text-main) !important;
        }

        /* Collapse button inside sidebar - override Streamlit's inline color attr */
        [data-testid="stSidebarCollapseButton"] button {
            color: var(--text-main) !important;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        [data-testid="stSidebarCollapseButton"] span[color],
        [data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"] {
            color: var(--text-main) !important;
        }

        /* collapsedControl tab lives OUTSIDE stSidebar - needs its own rules */
        [data-testid="collapsedControl"] {
            background: #f1f5f9 !important;
            border-right: 1px solid var(--surface-border) !important;
        }
        [data-testid="collapsedControl"] button {
            color: var(--text-main) !important;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        [data-testid="collapsedControl"] span[color],
        [data-testid="collapsedControl"] [data-testid="stIconMaterial"] {
            color: var(--text-main) !important;
        }

        /* ── Floating Radio Buttons ──────────────────────────────────────── */
        [data-testid="stRadio"] > div > label {
            color: rgba(0,0,0,0.5) !important; /* Changed to dark grey */
            font-size: 0.65rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.2em !important;
            margin-bottom: 8px !important;
        }
        [data-testid="stRadio"] label {
            color: var(--text-muted) !important;
            font-size: 0.88rem !important;
            letter-spacing: 0.05em !important;
            padding: 4px 0 !important;
        }
        [data-testid="stRadio"] label:has(input:checked) {
            color: #000000 !important; /* Changed to black */
            font-weight: 600 !important;
        }
        [data-testid="stRadio"] [data-baseweb="radio"] > div {
            background-color: transparent !important;
            border-color: var(--surface-border) !important;
        }
        [data-testid="stRadio"] label:has(input:checked) [data-baseweb="radio"] > div {
            background-color: var(--accent-primary) !important;
            border-color: var(--accent-primary) !important;
            box-shadow: none !important;
        }

        /* ── Main Input & Buttons (Solid Pills) ──────────────────────────── */
        [data-testid="stTextInput"] > div > div,
        .stTextInput > div > div,
        [data-baseweb="base-input"],
        [data-baseweb="input"] {
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
            overflow: visible !important;
            border-radius: 40px !important;
        }
        
        [data-testid="stTextInput"] input,
        .stTextInput input {
            background: var(--surface-bg) !important;
            color: var(--text-main) !important;
            border: 1px solid var(--surface-border) !important;
            border-radius: 40px !important; /* Pill shape */
            height: 52px !important;
            padding: 0 24px !important;
            font-family: 'Outfit', sans-serif !important;
            font-size: 1rem !important;
            font-weight: 400 !important;
            transition: all 0.2s ease !important;
            box-sizing: border-box !important;
        }
        [data-testid="stTextInput"] input::placeholder,
        .stTextInput input::placeholder {
            color: var(--text-muted) !important;
            opacity: 1 !important;
        }
        [data-testid="stTextInput"] input:focus,
        .stTextInput input:focus {
            border-color: #cbd5e1 !important;
            background: var(--surface-bg) !important;
            box-shadow: 0 0 16px rgba(0,0,0,0.05) !important;
            outline: none !important;
        }
        [data-testid="stTextInput"] label,
        .stTextInput label { display: none !important; }

        /* General & Form Submit Buttons - Sharp Yellow */
        .stButton > button,
        .stButton button,
        [data-testid="stFormSubmitButton"] > button,
        [data-testid="stFormSubmitButton"] button,
        button[kind="formSubmit"],
        button[kind="secondary"] {
            background: var(--accent-primary) !important;
            color: #000000 !important;
            border: none !important;
            border-radius: 40px !important; /* Pill shape */
            font-family: 'Outfit', sans-serif !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
            height: 52px !important;
            padding: 0 24px !important;
            line-height: 52px !important;
            transition: all 0.2s ease !important;
            width: 100% !important;
            box-sizing: border-box !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        .stButton > button:hover,
        [data-testid="stFormSubmitButton"] > button:hover,
        button[kind="formSubmit"]:hover,
        button[kind="secondary"]:hover {
            background: #bed400 !important;
            transform: scale(0.98);
            box-shadow: none !important;
        }
        .stButton > button:active,
        [data-testid="stFormSubmitButton"] > button:active {
            transform: scale(0.95);
        }

        /* ── File Uploader (Solid Dropzone) ────────────────────────────── */
        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploaderDropzone"] {
            background: var(--surface-bg) !important;
            border: 1px solid var(--surface-border) !important;
            border-radius: 20px !important;
            padding: 40px !important;
            transition: all 0.2s !important;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: #cbd5e1 !important;
            background: #f1f5f9 !important;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] span,
        [data-testid="stFileUploaderDropzoneInstructions"] small {
            color: rgba(255,255,255,0.5) !important;
        }

        /* ── Expanders (Sleek Foldout) ───────────────────────────────────── */
        [data-testid="stExpander"] {
            background: var(--surface-bg) !important;
            border: 1px solid var(--surface-border) !important;
            border-radius: 12px !important;
            margin-bottom: 12px !important;
            overflow: hidden !important;
            transition: all 0.2s !important;
        }
        [data-testid="stExpander"] summary {
            color: var(--text-main) !important;
            font-size: 0.9rem !important;
            padding: 16px 20px !important;
        }
        [data-testid="stExpander"] summary:hover { color: var(--text-main) !important; }
        [data-testid="stExpander"] summary svg { fill: var(--text-muted) !important; }

        /* ── Code Blocks (`pre` tags) ────────────────────────────────────── */
        pre, code {
            background: #f1f5f9 !important;
            border: 1px solid var(--surface-border) !important;
            border-radius: 8px !important;
            font-size: 0.8rem !important;
            color: #334155 !important;
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* ── General Layout & Typography ─────────────────────────────────── */
        h1, h2, h3, h4, h5, h6, [data-testid="stMarkdownContainer"] h3 {
            color: var(--text-main) !important;
        }

        hr {
            border: none !important;
            border-top: 1px solid var(--surface-border) !important;
            margin: 30px 0 !important;
        }
        .block-container {
            padding-top: 2rem !important;
            padding-left: 4rem !important;
            padding-right: 4rem !important;
            max-width: 1000px !important;
        }
        [data-testid="column"] { gap: 0 !important; }
        
        [data-testid="stSpinner"] > div {
            border-color: rgba(255,255,255,0.2) transparent transparent transparent !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PRIMITIVE WIDGETS
# ─────────────────────────────────────────────────────────────────────────────
def neon_header(text: str, colour: str = "#000") -> None:
    st.markdown(
        f"<h1 style='"
        f"color:{colour};"
        f"font-size:2rem;"
        f"font-weight:300;"
        f"letter-spacing:0.02em;"
        f"margin:0 0 24px 0;"
        f"padding-bottom:16px;"
        f"border-bottom:1px solid rgba(0,0,0,0.1);" # Changed border to dark
        f"animation: slideUpFade 0.6s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;"
        f"'>{text}</h1>",
        unsafe_allow_html=True,
    )


def small_label(text: str, colour: str = "rgba(0,0,0,0.6)") -> None: # Changed default text
    st.markdown(
        f"<p style='color:{colour};font-size:0.7rem;letter-spacing:0.15em;"
        f"text-transform:uppercase;margin:0 0 6px 0;font-weight:500;'>{text}</p>",
        unsafe_allow_html=True,
    )


def route_badge(route: str) -> None:
    c = ROUTE_COLOURS.get(route, "rgba(0,0,0,0.6)") # Changed fallback text
    st.markdown(
        f"<div style='animation: slideUpFade 0.8s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;'>"
        f"<span style='"
        f"display:inline-block;"
        f"background: rgba(0,0,0,0.05);" # Changed background to dark
        f"backdrop-filter: blur(8px);"
        f"color:{c};"
        f"border:1px solid {c};"
        f"border-radius:30px;"
        f"padding:4px 14px;"
        f"font-size:0.7rem;"
        f"letter-spacing:0.15em;"
        f"font-weight:500;"
        f"box-shadow:0 4px 12px rgba(0,0,0,0.05);" # Changed shadow
        f"'>{route.upper()}</span></div>",
        unsafe_allow_html=True,
    )

def answer_bubble(text: str) -> None:
    st.markdown(
        f"""<div style="
            background: var(--surface-bg);
            border: 1px solid var(--surface-border);
            border-left: 2px solid var(--accent-primary);
            border-radius: 0 16px 16px 16px;
            padding: 24px 28px;
            margin: 12px 0 24px 0;
            color: var(--text-main);
            font-size: 0.95rem;
            line-height: 1.7;
            font-weight: 300;
            white-space: pre-wrap;
            animation: slideUpFade 0.4s ease forwards;
        ">{text}</div>""",
        unsafe_allow_html=True,
    )


def question_bubble(text: str) -> None:
    st.markdown(
        f"""<div style="
            background: #f1f5f9;
            border: 1px solid var(--surface-border);
            border-right: 2px solid var(--accent-secondary);
            border-radius: 16px 0 16px 16px;
            padding: 16px 24px;
            margin: 24px 0 12px 60px;
            color: var(--text-main);
            font-size: 0.92rem;
            line-height: 1.6;
            font-weight: 400;
            text-align: right;
            animation: slideUpFade 0.3s ease forwards;
        ">{text}</div>""",
        unsafe_allow_html=True,
    )


# def route_badge(route: str) -> None:
#     c = ROUTE_COLOURS.get(route, "rgba(255,255,255,0.3)")
#     st.markdown(
#         f"<div style='animation: slideUpFade 0.8s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;'>"
#         f"<span style='"
#         f"display:inline-block;"
#         f"background: rgba(255,255,255,0.03);"
#         f"backdrop-filter: blur(8px);"
#         f"color:{c};"
#         f"border:1px solid {c};"
#         f"border-radius:30px;"
#         f"padding:4px 14px;"
#         f"font-size:0.7rem;"
#         f"letter-spacing:0.15em;"
#         f"font-weight:500;"
#         f"box-shadow:0 4px 12px rgba(0,0,0,0.1);"
#         f"'>{route.upper()}</span></div>",
#         unsafe_allow_html=True,
#     )


def error_card(message: str) -> None:
    st.markdown(
        f"<div style='"
        f"background:rgba(255, 60, 100, 0.1);"
        f"backdrop-filter: blur(10px);"
        f"border:1px solid rgba(255, 60, 100, 0.2);"
        f"border-radius:16px;"
        f"padding:16px 20px;"
        f"font-size:0.9rem;"
        f"color:#ff88aa;"
        f"font-weight:400;"
        f"margin:12px 0;"
        f"box-shadow: 0 8px 32px rgba(255, 60, 100, 0.05);"
        f"animation: slideUpFade 0.5s forwards;"
        f"'>{message}</div>",
        unsafe_allow_html=True,
    )


def success_card(message: str) -> None:
    st.markdown(
        f"<div style='"
        f"background:rgba(0, 255, 100, 0.05);"
        f"backdrop-filter: blur(10px);"
        f"border:1px solid rgba(0, 255, 100, 0.15);"
        f"border-radius:16px;"
        f"padding:16px 20px;"
        f"margin:12px 0;"
        f"font-size:0.9rem;"
        f"color:#88ffcc;"
        f"font-weight:400;"
        f"box-shadow: 0 8px 32px rgba(0, 255, 100, 0.05);"
        f"animation: slideUpFade 0.5s forwards;"
        f"'>{message}</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE RENDERING
# ─────────────────────────────────────────────────────────────────────────────
def render_image_chunk(chunk: dict) -> None:
    """Render an image chunk inside a sleek floating frame."""
    b64 = chunk.get("base64_image") or chunk.get("image_data")
    url = chunk.get("image_url")
    caption = f"page {chunk.get('page_number', '')} {chunk.get('section', '')}".strip()

    st.markdown(
        """<style>
        .stImage > img {
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.1);
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }
        .stImage > caption {
            color: rgba(255,255,255,0.4) !important;
            font-size: 0.75rem !important;
            margin-top: 8px !important;
        }
        </style>""",
        unsafe_allow_html=True
    )

    if b64:
        try:
            img_bytes = base64.b64decode(b64)
            st.image(img_bytes, use_container_width=False, caption=caption or None)
        except Exception:
            st.caption("[image decode error]")
    elif url:
        st.image(url, use_container_width=False, caption=caption or None)


def render_inline_images(sources: list) -> None:
    """Grid of image thumbnails rendered in a glassmorphic container."""
    image_chunks = [s for s in (sources or []) if s.get("chunk_type") == "image"]
    if not image_chunks:
        return

    st.markdown(
        "<div style='animation: slideUpFade 0.8s forwards;'>"
        "<p style='color:rgba(255,255,255,0.4);font-size:0.75rem;letter-spacing:0.15em;"
        "text-transform:uppercase;margin:16px 0 12px;font-weight:500;'>Visual Sources</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    cols_per_row = 3
    for i in range(0, len(image_chunks), cols_per_row):
        row = image_chunks[i : i + cols_per_row]
        cols = st.columns(len(row))
        for col, chunk in zip(cols, row):
            with col:
                render_image_chunk(chunk)
