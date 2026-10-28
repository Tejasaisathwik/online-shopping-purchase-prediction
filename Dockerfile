FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/tejasaisathwik/online-shopping-purchase-prediction"
LABEL org.opencontainers.image.description="Online Shopping Purchase Prediction API"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY online_shopping_model.joblib .
COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
