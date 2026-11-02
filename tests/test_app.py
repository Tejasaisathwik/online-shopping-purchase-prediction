"""Automated test suite for Online Shopping Purchase Prediction API.
Execute with: python -m pytest -q
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_and_log():
    sample_payload = {
        "Age": 35,
        "Gender": "Female",
        "MonthlyIncome": 75000.0,
        "WebsiteVisits": 12,
        "PagesViewed": 8,
        "TimeSpentMinutes": 25,
        "PreviousPurchases": 4,
        "CartItems": 3,
        "DiscountUsed": "No",
        "DeviceType": "Mobile",
        "Membership": "Basic",
        "CustomerRating": 4.2,
        "PurchaseAmount": 1500.0,
    }
    r = client.post("/predict", json=sample_payload)
    assert r.status_code == 200
    data = r.json()
    assert data["prediction"] in ("Purchased", "Not Purchased")
    assert "request_id" in data
    assert 0.0 <= data["prediction_confidence"] <= 1.0
    assert data["prediction_time_ms"] >= 0


def test_invalid_input():
    # CustomerRating cannot exceed 5.0 and Age cannot exceed 100
    invalid_payload = {
        "Age": 150,
        "Gender": "Female",
        "MonthlyIncome": 75000.0,
        "WebsiteVisits": 12,
        "PagesViewed": 8,
        "TimeSpentMinutes": 25,
        "PreviousPurchases": 4,
        "CartItems": 3,
        "DiscountUsed": "No",
        "DeviceType": "Mobile",
        "Membership": "Basic",
        "CustomerRating": 10.5,
        "PurchaseAmount": 1500.0,
    }
    r = client.post("/predict", json=invalid_payload)
    assert r.status_code == 422


def test_metrics():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "ml_http_requests_total" in r.text
    assert "ml_predictions_total" in r.text
