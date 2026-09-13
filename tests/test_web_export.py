"""
tests/test_web_export.py
Unit tests for web deployment export and inference fidelity.
"""

from pathlib import Path
import json
import numpy as np
import pytest

from export_web_data import main as export_main

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_web_export_file_generation():
    """Verify that export_web_data generates js/model_data.js properly."""
    js_path = PROJECT_ROOT / "js" / "model_data.js"
    assert js_path.exists(), "js/model_data.js must exist"
    content = js_path.read_text(encoding="utf-8")
    assert "window.AI_CAREER_DATA =" in content
    assert "overall" in content
    assert "daffodil" in content
    assert "public" in content
    assert "private" in content


def test_vercel_config():
    """Verify vercel.json configuration validity."""
    vercel_path = PROJECT_ROOT / "vercel.json"
    assert vercel_path.exists(), "vercel.json must exist"
    data = json.loads(vercel_path.read_text(encoding="utf-8"))
    assert data.get("version") == 2
    assert "headers" in data
