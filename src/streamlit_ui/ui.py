"""
ui.py
─────
Entry point — run with: streamlit run src/streamlit_ui/ui.py

Responsibilities:
  1. Page config (must be the very first Streamlit call)
  2. CSS injection
  3. Session state initialisation
  4. Sidebar: role picker + per-role actions
  5. Route to the correct view module
"""

import streamlit as st

from components import inject_css
import admin_view
import user_view
import dev_view

# ─────────────────────────────────────────────────────────────────────────────
# 1. PAGE CONFIG  (must come before any other st.* call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Banking",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
inject_css()

# ─────────────────────────────────────────────────────────────────────────────
# 3. SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "role" not in st.session_state:
    st.session_state.role = "User"
if "history_user" not in st.session_state:
    st.session_state.history_user = []   # {q: str, data: dict}
if "history_dev" not in st.session_state:
    st.session_state.history_dev = []

# ─────────────────────────────────────────────────────────────────────────────
# 4. SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        "<h1 style='color:#0f172a;font-size:1.1rem;font-weight:600;" 
        "letter-spacing:0.1em;margin-bottom:2px;'>Banking</h1>"
        "<p style='color:#303050;font-size:0.72rem;letter-spacing:0.06em;"
        "margin-top:0;margin-bottom:24px;'>RAG SYSTEM</p>",
        unsafe_allow_html=True,
    )

    selected_role = st.radio(
        "ROLE",
        ["User", "Dev", "Admin"],
        index=["User", "Dev", "Admin"].index(st.session_state.role),
    )

    # Clear both history lists whenever the role changes
    if selected_role != st.session_state.role:
        st.session_state.history_user = []
        st.session_state.history_dev  = []
        st.session_state.role = selected_role
        st.rerun()

    st.markdown("<hr/>", unsafe_allow_html=True)

    if selected_role in ("User", "Dev"):
        history_key = f"history_{selected_role.lower()}"
        if st.button("Clear history", key="clear_hist"):
            st.session_state[history_key] = []
            st.rerun()

    # st.markdown(
    #     "<p style='color:#252545;font-size:0.68rem;position:fixed;"
    #     "bottom:16px;letter-spacing:0.05em;'>v1.0</p>",
    #     unsafe_allow_html=True,
    # )



# ─────────────────────────────────────────────────────────────────────────────
# 5. ROUTE TO VIEW
# ─────────────────────────────────────────────────────────────────────────────
role = st.session_state.role

if role == "Admin":
    admin_view.render()
elif role == "User":
    user_view.render()
elif role == "Dev":
    dev_view.render()