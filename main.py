"""
api/main.py  —  FastAPI REST API for Fake News Detection inference.

Endpoints:
  POST /predict        — predict single text
  POST /predict/batch  — predict list of texts
  GET  /health         — health check
  GET  /model/info     — model metadata
"""

import os
import time
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from schemas import (
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse,
    HealthResponse, ModelInfoResponse,
)

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

# ──────────────────────────────────────────────────────────────
# App State
# ──────────────────────────────────────────────────────────────
predictor = None
model_load_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global predictor, model_load_time

    model_path = os.getenv("MODEL_PATH", "models/roberta-base")
    if Path(model_path).exists():
        try:
            from predict import FakeNewsPredictor
            t0 = time.time()
            predictor = FakeNewsPredictor(model_path)
            model_load_time = time.time() - t0
            print(f"✅ Model loaded in {model_load_time:.2f}s")
        except Exception as e:
            print(f"⚠️  Could not load model: {e}. Running in demo mode.")
    else:
        print(f"⚠️  Model not found at {model_path}. Running in demo mode.")

    yield
    predictor = None


# ──────────────────────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fake News Detection API",
    description="Detect fake news using fine-tuned transformer models (BERT/RoBERTa).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────
# Middleware: request timing
# ──────────────────────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{(time.time() - start) * 1000:.1f}ms"
    return response


# ──────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None,
        device=str(getattr(predictor, "device", "N/A")),
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    cfg = predictor.model.config
    return ModelInfoResponse(
        model_name=cfg.name_or_path,
        num_labels=cfg.num_labels,
        max_position_embeddings=cfg.max_position_embeddings,
        load_time_seconds=round(model_load_time or 0, 2),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict whether a news article is real or fake."""
    if predictor is None:
        # Demo mode: return mock prediction
        return _mock_prediction(request.text)

    try:
        result = predictor.predict(request.text, max_length=request.max_length)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict a batch of news articles."""
    if len(request.texts) > 32:
        raise HTTPException(status_code=400, detail="Max 32 texts per batch.")

    if predictor is None:
        results = [_mock_prediction(t) for t in request.texts]
        return BatchPredictionResponse(results=results, count=len(results))

    try:
        results = predictor.predict(request.texts, max_length=request.max_length)
        return BatchPredictionResponse(
            results=[PredictionResponse(**r) for r in results],
            count=len(results),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────
# Demo mode helper
# ──────────────────────────────────────────────────────────────
def _mock_prediction(text: str) -> PredictionResponse:
    """Return a deterministic mock prediction based on text heuristics."""
    text_lower = text.lower()
    fake_signals = ["shocking", "exposed", "conspiracy", "they don't want you", "secret", "coverup", "aliens"]
    is_fake = any(sig in text_lower for sig in fake_signals)
    label = "FAKE" if is_fake else "REAL"
    conf = 0.91 if is_fake else 0.87
    return PredictionResponse(
        text=text[:100],
        prediction=label,
        confidence=conf,
        emoji="🔴" if is_fake else "🟢",
        scores={"FAKE": conf if is_fake else 1 - conf, "REAL": 1 - conf if is_fake else conf},
    )
