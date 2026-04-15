import base64
import os
import pathlib

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.core.db import get_db_conn

load_dotenv()

_embeddings_model = GoogleGenerativeAIEmbeddings(
    model=os.getenv("GOOGLE_EMBEDDING_MODEL"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    output_dimensionality=1536,
)


def vector_search(query: str, k: int = 5, chunk_type: str | None = None) -> list[Document]:
    """Semantic vector similarity search using pgvector.

    Metadata keys mirror the raw DB column names exactly so downstream
    consumers (agents, run_agent sources) always see consistent field names:
      chunk_id, source_file, page_number, section, chunk_type,
      element_type, similarity, image_path, image_base64.
    """
    query_vec     = _embeddings_model.embed_query(query)
    embedding_str = "[" + ",".join(str(v) for v in query_vec) + "]"

    type_clause = "AND chunk_type = %(chunk_type)s" if chunk_type else ""

    sql = f"""
        SELECT
            id, content, chunk_type, page_number, section,
            source_file, element_type, image_path, mime_type,
            position, metadata,
            1 - (embedding <=> %(vec)s::vector) AS similarity
        FROM multimodal_chunks
        WHERE 1=1 {type_clause}
        ORDER BY embedding <=> %(vec)s::vector
        LIMIT %(k)s
    """

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"vec": embedding_str, "chunk_type": chunk_type, "k": k})
            rows = cur.fetchall()

    docs = []
    for row in rows:
        row = dict(row)

        img_path     = row.get("image_path")
        image_base64 = None
        if img_path and os.path.exists(img_path):
            image_base64 = base64.b64encode(
                pathlib.Path(img_path).read_bytes()
            ).decode()

        docs.append(
            Document(
                page_content=row["content"],
                metadata={
                    "chunk_id":     row.get("id"),
                    "source_file":  row.get("source_file", "unknown"),
                    "page_number":  row.get("page_number"),
                    "section":      row.get("section"),
                    "chunk_type":   row.get("chunk_type"),
                    "element_type": row.get("element_type"),
                    "similarity":   row.get("similarity"),
                    "image_path":   img_path,
                    "image_base64": image_base64,
                },
            )
        )

    return docs