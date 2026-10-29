import logging
import sys
import time
import uuid

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel

# Logging configuration (Practical-11)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

logger = logging.getLogger("online-shopping-api")

app = FastAPI(
    title="Online Shopping Purchase Prediction API",
    version="1.0.0",
)

# Prometheus metrics - Step 5 (Dataset & Practical-12 Rubric Compatibility)
HTTP_REQUESTS = Counter(
    "online_shopping_api_http_requests_total",
    "Total HTTP requests received by the Online Shopping API",
    ["method", "path", "status"],
)
HTTP_REQUESTS_RUBRIC = Counter(
    "student_api_http_requests_total",
    "Total HTTP requests received by the API (rubric alias)",
    ["method", "path", "status"],
)

HTTP_LATENCY = Histogram(
    "online_shopping_api_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)
HTTP_LATENCY_RUBRIC = Histogram(
    "student_api_http_request_duration_seconds",
    "HTTP request duration in seconds (rubric alias)",
    ["method", "path"],
)

PREDICTIONS = Counter(
    "online_shopping_api_predictions_total",
    "Total Online Shopping Prediction attempts",
    ["result"],
)
PREDICTIONS_RUBRIC = Counter(
    "student_api_predictions_total",
    "Total Prediction attempts (rubric alias)",
    ["result"],
)

MODEL_LOADED = Gauge(
    "online_shopping_api_model_loaded",
    "Whether the Online Shopping Prediction model is loaded",
)
MODEL_LOADED_RUBRIC = Gauge(
    "student_api_model_loaded",
    "Whether the model is loaded (rubric alias)",
)

MODEL_FILE = "online_shopping_model.joblib"

# Load trained model - Step 6
try:
    model = joblib.load(MODEL_FILE)
    MODEL_LOADED.set(1)
    MODEL_LOADED_RUBRIC.set(1)
    logger.info("event=model_loaded model_file=%s", MODEL_FILE)
except Exception:
    MODEL_LOADED.set(0)
    MODEL_LOADED_RUBRIC.set(0)
    logger.exception("event=model_load_failed model_file=%s", MODEL_FILE)
    raise


# Input data schema
class ShoppingInput(BaseModel):
    Age: int
    Gender: str
    MonthlyIncome: float
    WebsiteVisits: int
    PagesViewed: int
    TimeSpentMinutes: int
    PreviousPurchases: int
    CartItems: int
    DiscountUsed: str
    DeviceType: str
    Membership: str
    CustomerRating: float
    PurchaseAmount: float


# Request logging middleware - Step 7
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    path = request.url.path
    method = request.method
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
        duration = time.perf_counter() - start_time
        duration_ms = duration * 1000

        # Do not count Prometheus' own scrape requests in user-traffic metrics
        if path != "/metrics":
            HTTP_REQUESTS.labels(
                method=method,
                path=path,
                status=str(response.status_code),
            ).inc()
            HTTP_REQUESTS_RUBRIC.labels(
                method=method,
                path=path,
                status=str(response.status_code),
            ).inc()
            HTTP_LATENCY.labels(
                method=method,
                path=path,
            ).observe(duration)
            HTTP_LATENCY_RUBRIC.labels(
                method=method,
                path=path,
            ).observe(duration)

        level = (
            logging.WARNING
            if response.status_code >= 400
            else logging.INFO
        )

        logger.log(
            level,
            "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
            request_id,
            method,
            path,
            response.status_code,
            duration_ms,
        )

        response.headers["X-Request-ID"] = request_id
        return response

    except Exception:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.exception(
            "event=http_request_failed request_id=%s method=%s path=%s duration_ms=%.2f",
            request_id,
            method,
            path,
            duration_ms,
        )
        raise


# Root service endpoint
@app.get("/")
def root():
    return {
        "service": "Online Shopping Purchase Prediction API",
        "status": "ok",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "metrics": "/metrics",
            "docs": "/docs",
        },
    }


# Health check
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
    }


# Prediction endpoint - Step 8
@app.post("/predict")
def predict(data: ShoppingInput):
    try:
        payload = data.model_dump()
        logger.info(
            "event=prediction_request feature_names=%s",
            sorted(payload.keys()),
        )

        input_df = pd.DataFrame([payload])
        prediction = int(model.predict(input_df)[0])

        PREDICTIONS.labels(result="success").inc()
        PREDICTIONS_RUBRIC.labels(result="success").inc()

        logger.info(
            "event=prediction_success prediction=%s",
            str(prediction),
        )

        return {
            "prediction": prediction,
            "label": "Purchased" if prediction == 1 else "Not Purchased",
        }

    except Exception:
        PREDICTIONS.labels(result="error").inc()
        PREDICTIONS_RUBRIC.labels(result="error").inc()
        logger.exception("event=prediction_failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
        )


# Prometheus metrics endpoint - Step 9
@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
