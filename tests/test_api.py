"""
tests/test_api.py
Comprehensive Integration tests for FastAPI Vercel Serverless Function entrypoint.
Validates:
- Health and root checks
- Benchmarks retrieval
- Prediction across Overall, Public, Private, and Daffodil cohorts
- Both /api/* and /* route path compatibility
- Vercel 'from app import app' compatibility
"""

from fastapi.testclient import TestClient
import pytest

from api.index import app

client = TestClient(app)


def test_vercel_app_import():
    """Verify that 'from app import app' works for Vercel Python runtime entrypoint."""
    from app import app as vercel_app
    assert vercel_app is not None
    assert vercel_app.title == "AI Career Anxiety Prediction API"


def test_api_health_dual_routes():
    """Verify both /api/health and /health endpoints return HTTP 200."""
    for path in ["/api/health", "/health", "/api"]:
        response = client.get(path)
        assert response.status_code == 200, f"Failed on path {path}"
        data = response.json()
        assert data["status"] == "online"
        assert data["runtime"] == "Vercel Python Functions"
        assert "overall" in data["available_cohorts"]
        assert "daffodil" in data["available_cohorts"]


def test_api_root_serves_html_or_status():
    """Verify GET / returns index.html or fallback status."""
    response = client.get("/")
    assert response.status_code == 200


def test_api_benchmarks_dual_routes():
    """Verify frozen benchmark retrieval on /api/benchmarks and /benchmarks."""
    for path in ["/api/benchmarks", "/benchmarks"]:
        response = client.get(path)
        assert response.status_code == 200, f"Failed on path {path}"
        data = response.json()
        assert "frozen_benchmark" in data
        assert "Overall" in data["frozen_benchmark"]
        assert "Daffodil" in data["frozen_benchmark"]
        # Verify frozen research numbers are strictly preserved
        assert data["frozen_benchmark"]["Overall"]["test_f1"] == 0.7994
        assert data["frozen_benchmark"]["Daffodil"]["test_f1"] == 0.8019
        assert data["frozen_benchmark"]["Daffodil"]["test_roc_auc"] == 0.7418


@pytest.mark.parametrize("university,expected_cohort", [
    ("University of Dhaka", "public"),
    ("American International University-Bangladesh", "private"),
    ("Daffodil International University", "daffodil"),
    ("Other / Non-Surveyed Institution", "overall"),
])
def test_api_predict_cohorts(university, expected_cohort):
    """Verify predictions and SHAP attributions across all university cohorts."""
    payload = {
        "university": university,
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
    
    # Test both /api/predict and /predict
    for endpoint in ["/api/predict", "/predict"]:
        response = client.post(endpoint, json=payload)
        assert response.status_code == 200, f"Failed on {endpoint} for {university}: {response.text}"
        data = response.json()
        assert data["success"] is True
        assert data["cohort"] == expected_cohort
        assert data["prediction"]["predicted_class"] in [0, 1]
        assert 0.0 <= data["prediction"]["probability"] <= 1.0
        assert len(data["top_drivers"]) == 5
        assert len(data["all_features_ranked"]) == 17
        assert "engineered_features" in data
        assert len(data["engineered_features"]) == 17


def test_api_predict_invalid_data_handling():
    """Verify clean 422 error on invalid inputs instead of unhandled 500 crash."""
    payload = {
        "university": "University of Dhaka",
        "age": 999,  # violates le=35 validation
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422
