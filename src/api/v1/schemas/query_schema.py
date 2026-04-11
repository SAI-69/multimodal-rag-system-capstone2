from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class QueryRequest(BaseModel):
   query : str = Field(..., example="What is the minimum CIBIL score for personal loans?")

class QueryResponse(BaseModel):
   query: str = Field(description="User query")
   answer: str = Field(description="Generated answer")
   policy_citations: str = Field(description="Policy citations used")
   page_no: str = Field(description="Page numbers from documents")
   document_name: str = Field(description="Source document name")

class AIResponse(BaseModel):
   query: str = Field(description="The Given query by user must be present here")
   answer: str = Field(description="The generated response")
   policy_citations: str = Field(description="Give the Policy Citation")
   page_no: str = Field(description="The page number in the metadata")
   document_name: str = Field(description="Name of the document used")