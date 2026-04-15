import os
import asyncio
from typing import List
from fastapi import UploadFile
from src.ingestion.ingestion import run_ingestion  # <-- Your existing pipeline

TEMP_UPLOAD_DIR = "tmp_ingestion"
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

async def ingest_documents(files: List[UploadFile]) -> dict:
    total_chunks = 0
    processed = []
    errors = []

    for file in files:
        file_path = os.path.join(TEMP_UPLOAD_DIR, file.filename)
        try:
            # 1️⃣ Save uploaded file temporarily
            with open(file_path, "wb") as f:
                f.write(await file.read())

            # 2️⃣ Run your existing ingestion pipeline (in thread to avoid blocking FastAPI)
            result = await asyncio.to_thread(run_ingestion, file_path)
            total_chunks += result.get("chunks_ingested", 0)
            processed.append({
                "filename": file.filename,
                "doc_id": result.get("doc_id"),
                "status": result.get("status")
            })

        except Exception as e:
            errors.append({"filename": file.filename, "error": str(e)})
        finally:
            # 3️⃣ Cleanup temp file regardless of success/failure
            if os.path.exists(file_path):
                os.remove(file_path)

    return {
        "status": "partial_success" if errors else "success",
        "total_chunks_ingested": total_chunks,
        "files_processed": len(processed),
        "results": processed,
        "errors": errors
    }