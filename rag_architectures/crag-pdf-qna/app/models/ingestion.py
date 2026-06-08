from pydantic import BaseModel


class IngestedFile(BaseModel):
    filename: str
    chunks: int
    ids: list[str]


class FailedFile(BaseModel):
    filename: str
    error: str


class IngestResponse(BaseModel):
    files: list[IngestedFile]
    failed: list[FailedFile]
    total_chunks: int
