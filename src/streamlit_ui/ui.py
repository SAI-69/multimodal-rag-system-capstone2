import streamlit as st
import requests

API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(page_title="RAG System UI", layout="wide")

role = st.sidebar.radio(
    "Select Role",
    ["User", "Dev", "Admin"]
)

if role == "Admin":
    st.title("Admin Panel")

    upload_file = st.file_uploader("Upload PDF", type=["pdf"])

    if upload_file:
        with st.spinner("Uploading..."):
            try:
                files = {"file": (upload_file.name, upload_file.getvalue())}
                response = requests.post(f"{API_BASE}/admin/upload", files=files)

                if response.status_code == 200:
                    st.success("Upload successful")
                else:
                    st.error(response.text)
            except Exception as e:
                st.error(str(e))

elif role == "Dev":
    st.title("Dev Mode")

    query = st.text_input("Enter your query")

    if st.button("Get Answer"):
        with st.spinner("Fetching answer..."):
            response = requests.post(
                f"{API_BASE}/query",
                json={
                    "query": query,
                    "k": 5,
                    "chunk_type": None
                }
            )

            if response.status_code == 200:
                data = response.json()

                st.subheader("Answer")
                st.markdown(data["answer"])

                st.subheader("Retrieved Chunks")

                for r in data.get("sources", []):
                    with st.expander(
                        f"Page: {r.get('page_number')} | Type: {r.get('chunk_type')}"
                    ):
                        st.write(f"Section: {r.get('section')}")
                        st.write(f"Source: {r.get('source_file')}")
                        st.write(f"Similarity: {r.get('similarity')}")

            else:
                st.error(response.text)

elif role == "User":
    st.title("Ask Questions")

    query = st.text_input("Enter your query")

    if st.button("Get Answer"):
        with st.spinner("Fetching answer..."):
            response = requests.post(
                f"{API_BASE}/query",
                json={
                    "query": query,
                    "k": 5
                }
            )

            if response.status_code == 200:
                data = response.json()

                st.subheader("Answer")
                st.markdown(data["answer"])

            else:
                st.error(response.text)