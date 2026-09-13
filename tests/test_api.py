"""
tests/test_api.py
Integration tests for FastAPI Vercel Serverless Function entrypoint.
"""

from fastapi.testclient import TestClient
import pytest

from api.index import app

client = TestClient(app)


def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["runtime"] == "Vercel Python Functions"
    assert "overall" in data["available_cohorts"]


def test_api_benchmarks():
    response = client.get("/api/benchmarks")
    assert response.status_code == 200
    data = response.json()
    assert "frozen_benchmark" in data
    assert "Overall" in data["frozen_benchmark"]


def test_api_predict():
    payload = {
        "university": "University of Dhaka",
        "department": "Department of Computer Science and Engineering",
        "age": 22,
        "gender": "Male",
        "academic_year": "3rd Year",
        "ai_knowledge": "Medium",
        "ai_tools_used": "ChatGPT, Copilot",
        "career_path": "Software Engineer",
        "ai_tool_perception": "ChatGPT",
        "ai_future_perspective": "I think AI will replace some human jobs but also create new opportunities.",
        "ai_replace_jobs": "Partially",
        "ai_takeover_time": "6–10 years",
        "cohort_override": "auto",
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["cohort"] == "public"
    assert data["prediction"]["predicted_class"] in [0, 1]
    assert 0.0 <= data["prediction"]["probability"] <= 1.0
    assert len(data["top_drivers"]) > 0
