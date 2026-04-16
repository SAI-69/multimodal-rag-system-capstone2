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
    st.markdown("<div>", unsafe_allow_html=True)
    neon_header("Ingestion Protocol", "#0f172a")
    st.markdown(
        "<p style='color:rgba(0,0,0,0.6);font-size:0.95rem;font-weight:400; "
        "margin-bottom:30px;letter-spacing:0.02em;'> "
        "Upload verified PDF structural manifests into the vector knowledge base. </p>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # 1️⃣ Initialize processing state
    if "ingest_processing" not in st.session_state:
        st.session_state.ingest_processing = False

    # 2️⃣ File uploader (outside form for dynamic button control)
    uploaded_files = st.file_uploader(
        "Select one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="hidden",
    )

    # 3️⃣ Disable button if no files selected OR currently processing
    is_disabled = not uploaded_files or st.session_state.ingest_processing

    if st.button(
        "Initiate Ingestion", 
        disabled=is_disabled, 
        use_container_width=True, 
        key="ingest_btn"
    ):
        st.session_state.ingest_processing = True
        try:
            # 4️⃣ Spinner wraps the entire network call
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
                            f"Ingestion complete. <br><br> "
                            f"Files: {filenames} <br> "
                            f"Chunks ingested: {chunks} "
                        )
                        if errors:
                            msg += f"<br><br>⚠️ Errors: {[e.get('filename') for e in errors]} "
                        success_card(msg)
                    else:
                        error_card(f"System failure ({resp.status_code}): {resp.text}")
                except requests.exceptions.RequestException as exc:
                    error_card(f"Network error: {str(exc)}")
        finally:
            # 5️⃣ Reset state so button re-enables on next render
            st.session_state.ingest_processing = False
            # Streamlit automatically rerenders after script completion.
            # No explicit st.rerun() needed here (it would clear success/error cards).