from langchain_core.documents import Document

from src.api.v1.tools.fts_search import fts_search as _tool_fts_search
from src.api.v1.tools.vector_search import vector_search as _tool_vector_search


def hybrid_search(query: str, k: int = 5, rrf_k: int = 60) -> list[Document]:
    """Hybrid search combining vector similarity and FTS via Reciprocal Rank Fusion (RRF).

    Both retrieval methods are run independently with 2×k candidates each.
    Their results are fused using RRF scoring:

        score(d) = Σ  1 / (rrf_k + rank_in_list)

    where rrf_k=60 is the standard smoothing constant that prevents very
    high-ranked documents from dominating.  The top-k unique documents by
    combined RRF score are returned.

    Both sources return List[Document], so the fusion loop is uniform —
    doc identity is keyed on (source_file, page, page_content) since
    Document objects don't carry a DB id.

    Args:
        query: Natural-language search string.
        k:     Number of final results to return.
        rrf_k: RRF smoothing constant (default 60, per the original paper).

    Returns:
        List of LangChain Documents sorted by RRF score descending.
    """
    candidates = k * 2

    vector_docs = _tool_vector_search(query, k=candidates)
    fts_docs    = _tool_fts_search(query,    k=candidates)

    def _doc_key(doc: Document) -> str:
        return f"{doc.metadata.get('source_file')}|{doc.metadata.get('page_number')}|{doc.page_content[:80]}"

    scores:      dict[str, float]    = {}
    docs_by_key: dict[str, Document] = {}

    for rank, doc in enumerate(vector_docs, start=1):
        key = _doc_key(doc)
        scores[key]      = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
        docs_by_key[key] = doc

    for rank, doc in enumerate(fts_docs, start=1):
        key = _doc_key(doc)
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
        if key not in docs_by_key:
            docs_by_key[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)[:k]

    results = []
    for key in sorted_keys:
        doc = docs_by_key[key]
        doc.metadata["rrf_score"] = round(scores[key], 6)
        results.append(doc)

    return results