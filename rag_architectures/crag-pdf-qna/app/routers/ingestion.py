from fastapi import APIRouter, File, UploadFile
from app.models.ingestion import IngestResponse
from app.services.ingestion import ingest_uploads

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("", response_model=IngestResponse)
async def ingest(
    files: list[UploadFile] = File(..., description="One or more PDF files"),
) -> IngestResponse:
    """Ingest one or more PDFs: load -> sanitize -> chunk -> embed -> store."""
    succeeded, failed = await ingest_uploads(files)

    return IngestResponse(
        files=succeeded,
        failed=failed,
        total_chunks=sum(f.chunks for f in succeeded),
    )