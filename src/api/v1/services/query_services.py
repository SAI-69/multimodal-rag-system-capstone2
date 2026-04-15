from dotenv import load_dotenv

from src.api.v1.agents.agents import run_agent

load_dotenv(override=True)

def query_documents(query: str, k: int = 5, chunk_type: str | None = None) -> dict:

    """Invoke the LangGraph RAG agent and return the final answer with sources.

    The full retrieval → rerank → validate → generate → evaluate pipeline
    is handled inside the agent graph.  This service is now a thin adapter
    that maps the FastAPI request into the agent interface and returns a
    QueryResponse-compatible dict.

    Args:
        query:      The user's natural-language question.
        k:          Reserved for future per-request k tuning
                    (the agent uses its own default k=5 per tool).
        chunk_type: Reserved for future filtered retrieval support.

    Returns:
        dict with keys:
          answer  — the generated text answer (str)
          sources — list of source metadata dicts from the reranked documents
    """

    
    return run_agent(query)   # now returns full reference JSON
