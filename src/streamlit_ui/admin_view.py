"""
admin_view.py
─────────────
Admin role — upload PDF files to ingest.
"""

import requests
import streamlit as st

from components import (
    API_BASE,
    error_card,
    neon_header,
    success_card,
)


def render() -> None:
    st.markdown("<div style='animation: slideUpFade 0.4s forwards;'>", unsafe_allow_html=True)
    neon_header("Ingestion Protocol", "#0f172a")
    
    st.markdown(
        "<p style='color:rgba(0,0,0,0.6);font-size:0.95rem;font-weight:400;"
        "margin-bottom:30px;letter-spacing:0.02em;'>"
        "Upload verified PDF structural manifests into the vector knowledge base.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    with st.form("admin_ingest_form", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Select one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="hidden",
        )
        
        submitted = st.form_submit_button("Initiate Ingestion")

    if submitted:
        if not uploaded_files:
            error_card("Protocol violation: No data provided for ingestion.")
            return

        with st.spinner("Processing structural databanks..."):
            files_to_send = [
                ("files", (f.name, f.getvalue(), "application/pdf"))
                for f in uploaded_files
            ]
            try:
                resp = requests.post(
                    f"{API_BASE}/admin/upload",
                    files=files_to_send,
                    timeout=120,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    processed = data.get('results', [])
                    filenames = [r.get('filename', '?') for r in processed]
                    chunks = data.get('total_chunks_ingested', 0)
                    errors = data.get('errors', [])
                    msg = (
                        f"Ingestion complete.<br><br>"
                        f"Files: {filenames}<br>"
                        f"Chunks ingested: {chunks}"
                    )
                    if errors:
                        msg += f"<br><br>⚠️ Errors: {[e.get('filename') for e in errors]}"
                    success_card(msg)
                else:
                    error_card(f"System failure ({resp.status_code}): {resp.text}")
            except requests.exceptions.RequestException as exc:
                error_card(f"Network error: {str(exc)}")




