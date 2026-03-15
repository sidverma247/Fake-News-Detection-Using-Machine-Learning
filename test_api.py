"""tests/test_api.py — Integration tests for FastAPI endpoints."""

import sys
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "api"))

from main import app

client = TestClient(app)


class TestHealth:
    def test_health_ok(self):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data


class TestPredict:
    def test_real_news(self):
        r = client.post("/predict", json={"text": "Scientists publish new study on climate change impacts."})
        assert r.status_code == 200
        data = r.json()
        assert data["prediction"] in ("FAKE", "REAL")
        assert 0.0 <= data["confidence"] <= 1.0

    def test_fake_news(self):
        r = client.post("/predict", json={"text": "SHOCKING conspiracy exposed! They don't want you to know!"})
        assert r.status_code == 200
        data = r.json()
        assert data["prediction"] == "FAKE"

    def test_short_text_rejected(self):
        r = client.post("/predict", json={"text": "hi"})
        assert r.status_code == 422

    def test_response_schema(self):
        r = client.post("/predict", json={"text": "New government budget report released today."})
        data = r.json()
        assert "text" in data
        assert "prediction" in data
        assert "confidence" in data
        assert "scores" in data
        assert "FAKE" in data["scores"]
        assert "REAL" in data["scores"]


class TestBatchPredict:
    def test_batch(self):
        r = client.post("/predict/batch", json={
            "texts": [
                "Government releases annual budget.",
                "SHOCKING: Aliens land on Earth, media covering up!",
            ]
        })
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 2
        assert len(data["results"]) == 2

    def test_batch_too_large(self):
        r = client.post("/predict/batch", json={"texts": ["text"] * 33})
        assert r.status_code == 422  # validation error
