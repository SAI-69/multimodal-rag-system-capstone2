import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from src.api.v1.schemas.query_schema import QueryRequest, QueryResponse
from src.api.v1.services.query_services import query_documents
from src.api.v1.services.ingestion_service import ingest_documents

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    print(f"Received query: {request.query} with k={request.k}")
    result = query_documents(request.query, k=request.k, chunk_type=request.chunk_type)  
    return QueryResponse(**result)

@router.post("/admin/upload")
async def upload_endpoint(files: List[UploadFile] = File(..., description="PDF, DOCX, TXT, or MD files")):
    if not files:
        raise HTTPException(status_code=400, detail="At least one file must be uploaded.")

    # 🔒 Validate extensions before processing
    allowed_extensions = {".pdf", ".docx", ".txt", ".md"}
    for file in files:
        ext = os.path.splitext(file.filename)[1].lower() if file.filename else ""
        if ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {file.filename}")

    try:
        result = await ingest_documents(files)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion pipeline failed: {str(e)}")