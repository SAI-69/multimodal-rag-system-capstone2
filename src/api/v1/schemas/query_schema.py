from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ─── API Schemas ──────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., description="User query")
    k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    chunk_type: Optional[str] = Field(
        None, description="Filter by content type: 'text', 'table', or 'image'"
    )
class QueryResponse(BaseModel):
    answer: str = Field(description="Generated answer")
    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Source metadata for retrieved document chunks (vector search)"
    )
    sql_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata about SQL-based answer (tables, filters, row count)"
    )


class QueryResponse(BaseModel):
    answer: str = Field(description="Generated answer")

    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Source metadata for retrieved document chunks (vector search)"
    )

    sql_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata about SQL-based answer (tables, filters, row count)"
    )


# ─── Agent Schemas ────────────────────────────────────────────────────────────

class AIResponse(BaseModel):
    query: str = Field(description="The user's original query")
    answer: str = Field(description="The generated response")
    policy_citations: str = Field(description="Policy citations from the retrieved content")
    page_no: str = Field(description="Page number from the source document")
    document_name: str = Field(description="Name of the source document")


class FeedbackDetails(BaseModel):
    hallucination: str = Field(description="Feedback on hallucination")
    conciseness: str = Field(description="Feedback on conciseness")
    relevancy: str = Field(description="Feedback on relevancy")


class FeedBack(BaseModel):
    response: Literal["yes", "no"] = Field(description="'yes' if satisfactory, 'no' otherwise")
    feedback: FeedbackDetails