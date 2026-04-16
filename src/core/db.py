import base64
import hashlib
import json
import os
import pathlib

from dotenv import load_dotenv
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Connection setup
#
# The .env connection string uses SQLAlchemy's dialect prefix
# "postgresql+psycopg://" so that LangChain can parse it.
# psycopg.connect() expects the standard "postgresql://" URI, so we strip
# the dialect marker before passing it to psycopg.
# ---------------------------------------------------------------------------
_PG_CONNECTION = os.getenv("PG_CONNECTION_STRING", "")
_PG_DSN = _PG_CONNECTION.replace("postgresql+psycopg://", "postgresql://")

# How many chunks to embed per API call.
# Google's embedding API accepts up to 100 texts per batch.
_EMBED_BATCH_SIZE = 50

# ---------------------------------------------------------------------------
# Embeddings singleton (ingestion only)
#
# Used exclusively by store_chunks() to embed content at ingestion time.
# Search-time embedding is owned by tools/vector_search.py which has its
# own singleton, keeping db.py as a pure connection / storage layer.
# ---------------------------------------------------------------------------
_embeddings_model = GoogleGenerativeAIEmbeddings(
    model=os.getenv("GOOGLE_EMBEDDING_MODEL"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    output_dimensionality=1536,
)

# ---------------------------------------------------------------------------
# Issue 9 fix: Lazy connection pool — reuses existing TCP connections instead
# of opening a new one per request. Created on first use to avoid failing at
# import time when the DB is not yet available (e.g. during tests).
# ---------------------------------------------------------------------------
_pool: ConnectionPool | None = None


def _get_pool() -> ConnectionPool:
    """Return the module-level connection pool, creating it on first call."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            _PG_DSN,
            min_size=2,
            max_size=10,
            kwargs={"row_factory": dict_row},
        )
    return _pool


def get_db_conn():
    """Return a pooled connection context manager.

    Usage:
        with get_db_conn() as conn:
            with conn.cursor() as cur: ...
    """
    return _get_pool().connection()


# ---------------------------------------------------------------------------
# Document registry
# ---------------------------------------------------------------------------

def upsert_document(filename: str, source_path: str) -> str:
    """Insert a document record and return its UUID.

    Uses ON CONFLICT so re-ingesting the same filename updates the path
    and returns the *existing* doc_id rather than creating a duplicate.
    This makes ingestion idempotent at the document level.
    """
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (filename, source_path)
                VALUES (%s, %s)
                ON CONFLICT (filename) DO UPDATE
                    SET source_path = EXCLUDED.source_path,
                        ingested_at  = now()
                RETURNING id
                """,
                (filename, source_path),
            )
            row = cur.fetchone()
        conn.commit()
    return str(row["id"])


# ---------------------------------------------------------------------------
# Chunk storage
# ---------------------------------------------------------------------------

def store_chunks(chunks: list[dict], doc_id: str) -> int:
    """Embed each chunk and insert it into the multimodal_chunks table.

    Args:
        chunks:  List of dicts produced by parse_document() / ingestion.py.
                 Each dict must have: content (str), content_type (str),
                 metadata (dict with page_number, section, source_file,
                 element_type, position, image_base64).
        doc_id:  UUID string of the parent document (from upsert_document).

    Returns:
        Number of rows inserted.

    Embedding strategy:
        Texts are embedded in batches of _EMBED_BATCH_SIZE to minimise
        API round-trips. embed_documents() takes a list and returns a
        list of 768-dimensional float vectors in the same order.

    Vector storage:
        pgvector accepts the '[f1,f2,…]' string literal when cast with
        ::vector. We build that string directly to avoid needing the
        separate pgvector Python package.

    Image storage:
        image_base64 from metadata is decoded to raw bytes and stored in
        the BYTEA column. The JSONB metadata column does NOT duplicate it,
        keeping metadata lean.
    """
    if not chunks:
        return 0

    # ── Safely extract & guarantee strings for embedding ─────────────────────
    # Issue 8 fix: Ensure every item passed to embed_documents() is a string.
    # Handles cases where content might accidentally be a list, dict, or None.
    contents: list[str] = []
    for c in chunks:
        txt = c.get("content") if isinstance(c, dict) else c
        if txt is None:
            continue  # Skip empty chunks
        if isinstance(txt, str):
            contents.append(txt.strip())
        else:
            # Fallback: force-cast unexpected types (list, dict, etc.) to string
            contents.append(str(txt).strip())

    # ── Batch embed all chunks ────────────────────────────────────────────────
    all_embeddings: list[list[float]] = []
    
    for i in range(0, len(contents), _EMBED_BATCH_SIZE):
        batch = contents[i : i + _EMBED_BATCH_SIZE]
        
        if batch:
            # Ensure all items are definitely strings to prevent silent API drops
            clean_batch = [str(t).strip() if not isinstance(t, str) else t.strip() for t in batch]
            
            # Attempt to embed the batch
            batch_embeddings = _embeddings_model.embed_documents(clean_batch)
            
            # 🚨 SELF-HEALING FIX: If the API bugs out and returns a collapsed/truncated batch, 
            # fall back to embedding them individually using embed_query.
            if len(batch_embeddings) != len(clean_batch):
                print(f"⚠️ API batching bug detected (got {len(batch_embeddings)} embeddings for {len(clean_batch)} chunks). Falling back to singular embedding...")
                batch_embeddings = [_embeddings_model.embed_query(t) for t in clean_batch]
                
            all_embeddings.extend(batch_embeddings)

    # ── Insert rows ───────────────────────────────────────────────────────────
    # Issue 10 fix: Only store fields in JSONB that don't already have a
    # dedicated column — the rest are redundant and waste storage.
    _DEDICATED_COLUMNS = {
        "content_type", "element_type", "section",
        "page_number", "source_file", "position", "image_base64",
    }

    rows_inserted = 0
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            # Issue 4 fix: Delete stale chunks before re-inserting so that
            # re-ingesting the same document does not create duplicates.
            cur.execute(
                "DELETE FROM multimodal_chunks WHERE doc_id = %s::uuid",
                (doc_id,),
            )

            for chunk, embedding in zip(chunks, all_embeddings):
                meta = chunk["metadata"]

                # Issue 18 fix: Save image bytes to the filesystem and store
                # only the file path in the DB. This avoids bloating PostgreSQL
                # with large BYTEA columns that slow down vacuuming and queries.
                img_b64 = meta.get("image_base64")
                image_path: str | None = None
                mime_type = "image/png" if img_b64 else None
                if img_b64:
                    image_bytes = base64.b64decode(img_b64)
                    img_dir = pathlib.Path("data/images")
                    img_dir.mkdir(parents=True, exist_ok=True)
                    img_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
                    img_file = img_dir / f"{doc_id}_{img_hash}.png"
                    img_file.write_bytes(image_bytes)
                    image_path = str(img_file)

                # pgvector vector literal: '[0.1, 0.2, …]'
                embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

                # Exclude fields that already have dedicated columns from JSONB.
                clean_meta = {k: v for k, v in meta.items() if k not in _DEDICATED_COLUMNS}

                cur.execute(
                    """
                    INSERT INTO multimodal_chunks (
                        doc_id, chunk_type, element_type, content,
                        image_path, mime_type,
                        page_number, section, source_file,
                        position, embedding, metadata
                    ) VALUES (
                        %s::uuid, %s, %s, %s,
                        %s, %s,
                        %s, %s, %s,
                        %s::jsonb, %s::vector, %s::jsonb
                    )
                    """,
                    (
                        doc_id,
                        chunk["content_type"],       # chunk_type column
                        meta.get("element_type"),    # raw Docling label
                        chunk["content"],            # text / markdown / caption
                        image_path,                  # filesystem path (None for text/table)
                        mime_type,
                        meta.get("page_number"),
                        meta.get("section"),
                        meta.get("source_file"),
                        json.dumps(meta.get("position")) if meta.get("position") else None,
                        embedding_str,               # ::vector cast
                        json.dumps(clean_meta),      # JSONB catch-all
                    ),
                )
                rows_inserted += 1
        conn.commit()

    return rows_inserted


# ---------------------------------------------------------------------------
# Chunk listing (for preview / debugging)
# ---------------------------------------------------------------------------

def get_all_chunks(chunk_type: str | None = None, limit: int = 200) -> list[dict]:
    """Return all stored chunks, optionally filtered by type.

    Args:
        chunk_type: Optional filter — 'text', 'table', or 'image'.
        limit:      Max rows to return (default 200, safety cap).

    Returns:
        List of dicts with keys: id, content, chunk_type, page_number,
        section, source_file, element_type, image_base64, mime_type,
        position, metadata.
    """
    type_clause = "WHERE chunk_type = %(chunk_type)s" if chunk_type else ""

    sql = f"""
        SELECT
            id, content, chunk_type, page_number, section,
            source_file, element_type, image_path, mime_type,
            position, metadata
        FROM multimodal_chunks
        {type_clause}
        ORDER BY page_number ASC NULLS LAST, id ASC
        LIMIT %(limit)s
    """

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"chunk_type": chunk_type, "limit": limit})
            rows = cur.fetchall()

    results = []
    for row in rows:
        row = dict(row)
        img_path = row.pop("image_path", None)
        if img_path and os.path.exists(img_path):
            row["image_base64"] = base64.b64encode(
                pathlib.Path(img_path).read_bytes()
            ).decode()
        else:
            row["image_base64"] = None
        results.append(row)

    return results