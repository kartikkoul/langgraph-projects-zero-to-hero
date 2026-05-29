from typing import Literal

from pydantic import BaseModel, Field


class IntentSchema(BaseModel):
    intent: Literal["billing_agent", "technical_agent", "general_agent"]


class SentimentSchema(BaseModel):
    sentiment: str

class UrgencySchema(BaseModel):
    urgency: int = Field(..., ge=1, le=10)

class QualityScoreSchema(BaseModel):
    quality_score: int = Field(..., ge=1, le=10)