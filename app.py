import logging
import sys
import time
import uuid

import joblib
import pandas as pd

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger("online-shopping-api")

app = FastAPI(
    title="Online Shopping Purchase Prediction API"
)

MODEL_FILE = "online_shopping_model.joblib"

# Load trained model
try:
    model = joblib.load(MODEL_FILE)
    logger.info("event=model_loaded model_file=%s", MODEL_FILE)
except Exception:
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


# Request logging middleware
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get(
        "X-Request-ID",
        str(uuid.uuid4())
    )

    start = time.perf_counter()

    try:
        response = await call_next(request)

        duration_ms = (time.perf_counter() - start) * 1000

        level = (
            logging.WARNING
            if response.status_code >= 400
            else logging.INFO
        )

        logger.log(
            level,
            "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms
        )

        response.headers["X-Request-ID"] = request_id

        return response

    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000

        logger.exception(
            "event=http_request_failed request_id=%s method=%s path=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            duration_ms
        )

        raise


# Health check
@app.get("/health")
def health():
    return {"status": "healthy"}


# Prediction endpoint
@app.post("/predict")
def predict(data: ShoppingInput):
    try:
        payload = data.model_dump()

        logger.info(
            "event=prediction_request feature_names=%s",
            sorted(payload.keys())
        )

        input_df = pd.DataFrame([payload])

        prediction = model.predict(input_df)[0]

        logger.info(
            "event=prediction_success prediction=%s",
            str(prediction)
        )

        return {
            "prediction": int(prediction)
        }

    except Exception:
        logger.exception("event=prediction_failed")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )