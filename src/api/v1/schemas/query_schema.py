from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal


class QueryRequest(BaseModel):
   query : str = Field(..., example="What is the minimum CIBIL score for personal loans?")

class QueryResponse(BaseModel):
   #query: str = Field(description="User query")
   answer: str = Field(description="Generated answer")
   sources: List[str] = Field(description="Sources for the output generated.")

class AIResponse(BaseModel):
   query: str = Field(description="The Given query by user must be present here")
   answer: str = Field(description="The generated response")
   policy_citations: str = Field(description="Give the Policy Citation")
   page_no: str = Field(description="The page number in the metadata")
   document_name: str = Field(description="Name of the document used")

class FeedbackDetails(BaseModel):
    hallucination: str = Field(description="Feedback about hallucination")
    conciseness: str = Field(description="Feedback about conciseness")
    relevancy: str = Field(description="Feedback about relevancy")

class FeedBack(BaseModel):
    response: Literal["yes", "no"] = Field(description="Yes or No based on the feedback")
    feedback: FeedbackDetails