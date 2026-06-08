import asyncio
import logging
from fastapi import HTTPException, UploadFile
from app.models.ingestion import (
    FailedFile,
    IngestedFile,
)
from crag_agent.agent_api_layer import ingest_pdf_bytes
from crag_agent.agent.schemas import StandardError


MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB per file

async def _ingest_upload(file: UploadFile) -> IngestedFile:
    try: 
        if file.content_type not in ("application/pdf", "application/octet-stream"):
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type for '{file.filename}': {file.content_type}",
            )

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"Empty file: '{file.filename}'")
        if len(content) > MAX_FILE_BYTES:
            raise HTTPException(status_code=413, detail=f"File too large: '{file.filename}'")

        ids = await ingest_pdf_bytes(content=content, filename=file.filename) or []

        return IngestedFile(filename=file.filename, chunks=len(ids), ids=ids)

    except Exception as e:
        logging.error(f"Ingestion failed: {str(e)}")

        if isinstance(e, HTTPException):
            raise e

        raise HTTPException(status_code=500, detail=StandardError(
            message = "Internal server error",
            error = str(e),
            code = 500
        ))



async def ingest_uploads(
    files: list[UploadFile],
) -> tuple[list[IngestedFile], list[FailedFile]]:
    """Ingest many uploads concurrently, partitioning successes from failures.

    Uses return_exceptions=True so that one bad file never aborts the batch.
    """

    tasks = [_ingest_upload(file) for file in files]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    succeeded: list[IngestedFile] = []
    failed: list[FailedFile] = []
    for file, result in zip(files, results):
        if isinstance(result, IngestedFile):
            succeeded.append(result)
        elif isinstance(result, HTTPException):
            failed.append(FailedFile(filename=file.filename, error=str(result.detail)))
        elif isinstance(result, Exception):
            failed.append(FailedFile(filename=file.filename, error=str(result)))

    return succeeded, failed