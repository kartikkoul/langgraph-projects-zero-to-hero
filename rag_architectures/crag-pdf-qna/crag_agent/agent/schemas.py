from typing import Optional
from pydantic import BaseModel, Field


class StandardError(BaseModel):
    message: str
    error: Optional[str] = None
    code: Optional[int] = None

class DocEvalScore(BaseModel):
    score: float = Field(..., ge=0, le=1, description="Give a relevance score for the doc in the range from 0.0 to 1.0, where 0.0 is the least relevant and 1.0 is the most relevant. Example: 0.1, 0.5, 0.8")
    reason: str = Field(..., description="Reason for the score given.")

class KeepOrDropStrip(BaseModel):
    keep: bool = Field(..., description="Whether to keep the strip or drop it. Example: True, False")
    reason: str = Field(..., description="Reason for the decision to keep or drop the strip.")