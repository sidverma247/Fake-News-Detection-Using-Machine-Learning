"""api/schemas.py — Pydantic request/response models."""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=10000,
                      example="Scientists discover new renewable energy source.")
    max_length: int = Field(default=512, ge=64, le=512)


class PredictionResponse(BaseModel):
    text: str
    prediction: str            # "FAKE" or "REAL"
    confidence: float          # 0.0 – 1.0
    emoji: str                 # 🔴 / 🟢
    scores: Dict[str, float]   # {"FAKE": 0.12, "REAL": 0.88}


class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=32)
    max_length: int = Field(default=512, ge=64, le=512)


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


class ModelInfoResponse(BaseModel):
    model_name: str
    num_labels: int
    max_position_embeddings: int
    load_time_seconds: float
