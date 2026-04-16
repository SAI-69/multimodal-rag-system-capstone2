import base64
import os
import pathlib
import re
from dotenv import load_dotenv
from langchain_core.documents import Document
from src.core.db import get_db_conn

load_dotenv()

def fts_search(query: str, k: int = 5) -> list[Document]:
    """Production FTS: Query-aware scoring for acronyms & definitions."""
    
    query_words = query.strip().split()
    is_short_query = len(query_words) <= 3 or bool(re.search(r'\b[A-Z]{2,4}\b', query))
    
    acronym_pattern = r'\([A-Z]{2,5}\)'  
    
    ril_match = re.search(r'\b([A-Z]{2,5})\b', query.upper())
    ril_term = ril_match.group(1).lower() if ril_match else None
    
    table_boost = 1.0 if is_short_query else 2.0
    header_boost = 0.5 if is_short_query else 1.5

    sql = """
        SELECT 
            mc.id, mc.content, mc.chunk_type, mc.page_number, mc.section, 
            mc.element_type, mc.image_path, mc.metadata, d.filename AS source_file,
            
            (
                -- 1. Exact phrase match (Highest Priority)
                CASE WHEN to_tsvector('english', mc.content) @@ phraseto_tsquery('english', %(query)s::text) 
                     THEN 10.0 ELSE 0.0 END +
                     
                -- 2. All query words present (Medium Priority)
                CASE WHEN to_tsvector('english', mc.content) @@ websearch_to_tsquery('english', %(query)s::text) 
                     THEN 5.0 ELSE 0.0 END +
                     
                -- 3. Per-word match (Base Priority - prevents 0 results)
                (SELECT COUNT(*) FROM unnest(string_to_array(%(query)s::text, ' ')) AS word 
                 WHERE word != '' AND mc.content ILIKE '%%' || word || '%%') * 1.0 +
                 
                -- 4. DEFINITION PATTERN BOOST: parenthetical acronyms like "(RIL)"
                CASE WHEN mc.content ~* %(acronym_pattern)s::text THEN 3.0 ELSE 0.0 END +
                
                -- 5. CO-OCCURRENCE BOOST: RIL + Reliance in same chunk (strong definition signal)
                CASE 
                    WHEN %(ril_term)s::text IS NOT NULL 
                         AND mc.content ILIKE '%%' || %(ril_term)s::text || '%%' 
                         AND mc.content ILIKE '%%reliance%%' 
                    THEN 4.0 ELSE 0.0 END +
                 
                -- 6. Structural boosts (CONDITIONAL: reduced for short queries)
                CASE 
                    WHEN mc.chunk_type = 'table' THEN %(table_boost)s
                    WHEN mc.element_type = 'section_header' THEN %(header_boost)s
                    ELSE 0.0 
                END
            ) AS similarity
            
        FROM multimodal_chunks mc
        JOIN documents d ON mc.doc_id = d.id
        
        WHERE 
            -- Retrieve if phrase matches, OR all words match, OR ANY single word matches
            to_tsvector('english', mc.content) @@ phraseto_tsquery('english', %(query)s::text)
            OR to_tsvector('english', mc.content) @@ websearch_to_tsquery('english', %(query)s::text)
            OR EXISTS (
                SELECT 1 FROM unnest(string_to_array(%(query)s::text, ' ')) AS word
                WHERE word != '' AND mc.content ILIKE '%%' || word || '%%'
            )
            
        ORDER BY similarity DESC
        LIMIT %(k)s;
    """

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {
                "query": query.strip(),
                "k": k,
                "acronym_pattern": acronym_pattern,
                "ril_term": ril_term,
                "table_boost": table_boost,
                "header_boost": header_boost,
            })
            rows = cur.fetchall()

    docs = []
    for row in rows:
        row = dict(row)
        img_path = row.get("image_path")
        image_base64 = None
        if img_path and os.path.exists(img_path):
            image_base64 = base64.b64encode(pathlib.Path(img_path).read_bytes()).decode()

        metadata = row.get("metadata", {}) or {}
        metadata.update({
            "chunk_id": row.get("id"),
            "source_file": row.get("source_file", "unknown"),
            "page_number": row.get("page_number"),
            "section": row.get("section"),
            "chunk_type": row.get("chunk_type"),
            "element_type": row.get("element_type"),
            "similarity": float(row.get("similarity", 0.0)),
            "image_path": img_path,
            "image_base64": image_base64,
        })

        docs.append(Document(page_content=row["content"], metadata=metadata))

    print(f"[fts_search] Returned {len(docs)} docs for query: '{query}'")
    if docs:
        top = docs[0]
        content_preview = top.page_content[:150].replace('\n', ' ')
        print(f"   → Top chunk → Score: {top.metadata['similarity']:.2f} | "
              f"Page {top.metadata.get('page_number')} | Type: {top.metadata.get('chunk_type')}")
        print(f"   → Preview: '{content_preview}...'")
        if ril_term and ril_term in top.page_content.lower() and 'reliance' in top.page_content.lower():
            print(f"   → ✨ Co-occurrence boost applied: '{ril_term}' + 'Reliance'")
        if re.search(acronym_pattern, top.page_content):
            print(f"   → ✨ Definition pattern boost applied: found '(XYZ)' format")

    print("-" * 80)
    print(f"fts docs : {docs}")
              
    return docs