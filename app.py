"""Monitored FastAPI application for Online Shopping Purchase Prediction.
Integrates Pydantic schema validation, JSONL event logging, and Prometheus metrics.
"""
import json
import logging
from pathlib import Path
import time
import uuid
from datetime import datetime, timezone
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, Response
import joblib
import pandas as pd
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
ARTIFACT_PATHS = [
    ROOT / "models" / "online_shopping_model.joblib",
    ROOT / "models" / "student_result.joblib",
    ROOT / "online_shopping_model.joblib",
]

artifact_file = None
for p in ARTIFACT_PATHS:
    if p.exists():
        artifact_file = p
        break

if not artifact_file:
    raise RuntimeError("Model artifact not found. Please run python train_model.py first.")

saved = joblib.load(artifact_file)
if isinstance(saved, dict):
    model = saved["model"]
    features = saved.get("features", [])
    version = saved.get("version", "demo-v1")
else:
    model = saved
    features = [
        "Age", "Gender", "MonthlyIncome", "WebsiteVisits", "PagesViewed",
        "TimeSpentMinutes", "PreviousPurchases", "CartItems", "DiscountUsed",
        "DeviceType", "Membership", "CustomerRating", "PurchaseAmount"
    ]
    version = "demo-v1"

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Prometheus metrics matching the manual exactly
HTTP_REQUESTS = Counter(
    "ml_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)
HTTP_DURATION = Histogram(
    "ml_http_request_duration_seconds",
    "HTTP latency in seconds",
    ["path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)
PREDICTIONS = Counter(
    "ml_predictions_total",
    "Model predictions",
    ["predicted_label", "model_version"],
)

# Rubric and dataset alias metrics for seamless backwards compatibility
HTTP_REQUESTS_ALIAS = Counter(
    "online_shopping_api_http_requests_total",
    "Total HTTP requests received (alias)",
    ["method", "path", "status"],
)
HTTP_LATENCY_ALIAS = Histogram(
    "online_shopping_api_http_request_duration_seconds",
    "HTTP request latency (alias)",
    ["path"],
)
PREDICTIONS_ALIAS = Counter(
    "online_shopping_api_predictions_total",
    "Model predictions (alias)",
    ["result"],
)
MODEL_LOADED = Gauge(
    "online_shopping_api_model_loaded",
    "Model loaded indicator (1=ready, 0=not loaded)",
)
MODEL_LOADED.set(1)

# Configure JSONL event logging
logger = logging.getLogger("online_shopping_predictions")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = logging.FileHandler(LOG_DIR / "predictions.jsonl", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

app = FastAPI(
    title="Online Shopping Purchase Prediction - MLOps Monitoring Lab",
    description="Monitored FastAPI service exposing predictions, structured JSONL logs, and Prometheus metrics.",
    version="1.0",
)


class ShoppingFeatures(BaseModel):
    Age: int = Field(..., ge=18, le=100, description="Customer age (18 to 100)")
    Gender: Literal["Male", "Female"] = Field(..., description="Customer gender")
    MonthlyIncome: float = Field(..., ge=0, description="Monthly income in USD")
    WebsiteVisits: int = Field(..., ge=0, le=100, description="Total website visits in session")
    PagesViewed: int = Field(..., ge=0, le=100, description="Total pages viewed in session")
    TimeSpentMinutes: int = Field(..., ge=0, le=500, description="Minutes spent on site")
    PreviousPurchases: int = Field(..., ge=0, le=100, description="Number of past purchases")
    CartItems: int = Field(..., ge=0, le=50, description="Number of items added to cart")
    DiscountUsed: Literal["Yes", "No"] = Field(..., description="Whether a discount coupon was applied")
    DeviceType: Literal["Mobile", "Desktop", "Tablet"] = Field(..., description="Device used for browsing")
    Membership: Literal["Basic", "Premium", "Gold"] = Field(..., description="Membership tier")
    CustomerRating: float = Field(..., ge=1.0, le=5.0, description="Customer rating (1.0 to 5.0)")
    PurchaseAmount: float = Field(..., ge=0, description="Total order purchase amount")


@app.middleware("http")
async def track_http(request: Request, call_next):
    started = time.perf_counter()
    path = request.url.path if request.url.path in {"/", "/health", "/predict"} else "other"
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        status = 500
        raise
    finally:
        if request.url.path != "/metrics":
            duration = time.perf_counter() - started
            HTTP_REQUESTS.labels(method=request.method, path=path, status=str(status)).inc()
            HTTP_REQUESTS_ALIAS.labels(method=request.method, path=path, status=str(status)).inc()
            HTTP_DURATION.labels(path=path).observe(duration)
            HTTP_LATENCY_ALIAS.labels(path=path).observe(duration)
    return response


@app.get("/")
def home():
    return {
        "service": "Online Shopping Purchase Prediction Monitoring API",
        "status": "ready",
        "docs": "/docs",
        "metrics": "/metrics",
        "health": "/health",
        "model_version": version,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_version": version,
        "model_loaded": True,
    }


@app.post("/predict")
def predict(customer: ShoppingFeatures):
    start = time.perf_counter()
    values = customer.model_dump()
    row = pd.DataFrame([values])

    try:
        prediction_val = int(model.predict(row)[0])
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(row)[0, prediction_val])
        else:
            probability = 1.0
    except Exception as exc:
        PREDICTIONS_ALIAS.labels(result="error").inc()
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    label = "Purchased" if prediction_val == 1 else "Not Purchased"
    request_id = str(uuid.uuid4())

    # Prometheus metric increments
    PREDICTIONS.labels(predicted_label=label, model_version=version).inc()
    PREDICTIONS_ALIAS.labels(result="success").inc()

    # Structured JSONL event logging
    log_event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "model_version": version,
        **values,
        "predicted_label": label,
        "prediction_confidence": round(probability, 4),
        "prediction_time_ms": latency_ms,
    }
    logger.info(json.dumps(log_event, ensure_ascii=False))

    return {
        "request_id": request_id,
        "prediction": label,
        "predicted_label": label,
        "prediction_confidence": round(probability, 4),
        "prediction_time_ms": latency_ms,
        "model_version": version,
    }


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
